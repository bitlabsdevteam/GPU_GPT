"""
Lesson 06: ZeRO-3 semantics via PyTorch FSDP FULL_SHARD for GPT-style training.

Run (example, 4 GPUs):
  torchrun --nproc_per_node=4 lessons/lesson_06_fsdp_zero3_gpt2.py

What this demonstrates:
- ZeRO-3-style memory behavior (params + grads + optimizer states sharded)
- Transformer auto-wrap for communication/computation granularity control
- BF16 mixed precision policy for practical throughput
- Rank-0 full checkpoint save with CPU offload to avoid save-time OOM
"""

import os
import time
import statistics
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass
class TrainConfig:
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    seq_len: int = 256
    batch_size: int = 8
    n_samples: int = 16_384
    max_steps: int = 120
    grad_accum_steps: int = 4
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    use_bf16: bool = True


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok(input_ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


def setup_dist() -> tuple[int, int, int, torch.device]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device(f"cuda:{local_rank}")


def mean_all_ranks(x: float, device: torch.device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def build_dataset(cfg: TrainConfig) -> TensorDataset:
    x = torch.randint(0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    return TensorDataset(x, y)


def main() -> None:
    cfg = TrainConfig()
    rank, world, local_rank, device = setup_dist()

    torch.manual_seed(100 + rank)
    torch.cuda.manual_seed_all(100 + rank)

    model = TinyGPT(cfg).to(device)

    mp_policy = None
    if cfg.use_bf16:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

    auto_wrap = transformer_auto_wrap_policy
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        forward_prefetch=False,
        use_orig_params=True,
    )

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    ds = build_dataset(cfg)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    torch.cuda.reset_peak_memory_stats(device)
    fsdp_model.train()
    step_times_ms: list[float] = []
    loader_iter = iter(loader)

    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        for micro in range(cfg.grad_accum_steps):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                sampler.set_epoch(step)
                loader_iter = iter(loader)
                inp, tgt = next(loader_iter)

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                logits = fsdp_model(inp)
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    tgt[:, :-1].contiguous().view(-1),
                )
                loss = loss / cfg.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(device)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms.append(dt_ms)

        if step % 10 == 0:
            avg_loss = mean_all_ranks(float(loss.item() * cfg.grad_accum_steps), device)
            avg_ms = mean_all_ranks(dt_ms, device)
            if rank == 0:
                print(f"step={step:03d} loss={avg_loss:.4f} step_ms={avg_ms:.1f}")

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    mean_ms_local = statistics.mean(step_times_ms[-20:]) if len(step_times_ms) >= 20 else statistics.mean(step_times_ms)
    mean_ms = mean_all_ranks(mean_ms_local, device)
    peak_mem_mb = mean_all_ranks(peak_mem_mb, device)

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        full_state = fsdp_model.state_dict()

    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        out = "checkpoints/lesson06_fsdp_zero3_style.pt"
        torch.save(
            {
                "model": full_state,
                "config": cfg.__dict__,
                "world_size": world,
                "metrics": {
                    "mean_step_ms_last_window": mean_ms,
                    "mean_peak_mem_mb": peak_mem_mb,
                },
            },
            out,
        )
        print(f"saved {out}")
        print(f"summary: mean_step_ms={mean_ms:.1f} mean_peak_mem_mb={peak_mem_mb:.1f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
