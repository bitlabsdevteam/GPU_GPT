"""
Lesson 05: ZeRO-2 with GPT-style training (PyTorch DDP + ZeRO optimizer-state sharding + gradient bucket views)

Run (example, 4 GPUs):
  torchrun --nproc_per_node=4 lessons/lesson_05_zero2_ddp_grad_sharding.py

What this lesson demonstrates:
- ZeRO-1 via ZeroRedundancyOptimizer (optimizer-state sharding)
- Memory-aware gradient behavior using DDP gradient-as-bucket-view
- Gradient accumulation + no_sync to reduce communication frequency
- Instrumentation for step time and peak memory

Notes:
- Native PyTorch does not expose a one-flag "ZeRO-2" exactly like DeepSpeed/FSDP,
  but you can capture much of the practical behavior via:
    (1) optimizer state sharding (ZeRO-1), and
    (2) gradient bucket views / comm-aware accumulation.
- For full parameter+gradient+optimizer partitioning semantics, compare FSDP/DeepSpeed.
"""

import os
import time
import statistics
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


@dataclass
class TrainConfig:
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
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
    device = torch.device(f"cuda:{local_rank}")
    return rank, world, local_rank, device


def build_dataset(cfg: TrainConfig) -> TensorDataset:
    x = torch.randint(0, cfg.vocab_size, (cfg.n_samples, cfg.seq_len), dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    return TensorDataset(x, y)


def mean_all_ranks(x: float, device: torch.device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def main() -> None:
    cfg = TrainConfig()
    rank, world, local_rank, device = setup_dist()

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)

    model = TinyGPT(cfg).to(device)

    # gradient_as_bucket_view=True reduces grad memory overhead by viewing gradients
    # directly into all-reduce buckets (important practical ZeRO-2-style behavior).
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        static_graph=False,
    )

    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    ds = build_dataset(cfg)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    step_times_ms: list[float] = []
    ddp_model.train()
    loader_iter = iter(loader)

    torch.cuda.reset_peak_memory_stats(device)

    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()

        for micro in range(cfg.grad_accum_steps):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                sampler.set_epoch(step)
                loader_iter = iter(loader)
                inp, tgt = next(loader_iter)

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            # no_sync on all but final micro-step cuts all-reduce frequency.
            sync_ctx = ddp_model.no_sync() if micro < (cfg.grad_accum_steps - 1) else torch.enable_grad()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                    logits = ddp_model(inp)
                    loss = nn.functional.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        tgt[:, :-1].contiguous().view(-1),
                    )
                    loss = loss / cfg.grad_accum_steps
                loss.backward()

        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        step_times_ms.append(elapsed_ms)

        if step % 10 == 0:
            avg_loss = mean_all_ranks(float(loss.item() * cfg.grad_accum_steps), device)
            avg_ms = mean_all_ranks(elapsed_ms, device)
            if rank == 0:
                print(f"step={step:03d} loss={avg_loss:.4f} step_ms={avg_ms:.1f}")

    # Consolidate ZeRO sharded optimizer state before rank-0 checkpoint save.
    if rank == 0:
        optimizer.consolidate_state_dict(to=0)
    dist.barrier()

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    mean_ms = statistics.mean(step_times_ms[-20:]) if len(step_times_ms) >= 20 else statistics.mean(step_times_ms)
    mean_ms = mean_all_ranks(mean_ms, device)
    peak_mem_mb = mean_all_ranks(peak_mem_mb, device)

    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "model": ddp_model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__,
            "world_size": world,
            "metrics": {
                "mean_step_ms_last_window": mean_ms,
                "mean_peak_mem_mb": peak_mem_mb,
            },
        }
        out = "checkpoints/lesson05_zero2_style.pt"
        torch.save(ckpt, out)
        print(f"saved {out}")
        print(f"summary: mean_step_ms={mean_ms:.1f} mean_peak_mem_mb={peak_mem_mb:.1f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
