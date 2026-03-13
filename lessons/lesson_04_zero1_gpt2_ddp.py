"""
Lesson 04: ZeRO-1 with GPT-2 + DDP (PyTorch)

Run (example, 4 GPUs):
  torchrun --nproc_per_node=4 lessons/lesson_04_zero1_gpt2_ddp.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_dist() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


class TinyGPT2Block(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8):
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


class TinyGPT2(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([TinyGPT2Block(d_model=d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok(input_ids) + self.pos(positions)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


def build_fake_data(n_samples: int = 8192, seq_len: int = 128, vocab_size: int = 50257):
    x = torch.randint(0, vocab_size, (n_samples, seq_len), dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    ds = TensorDataset(x, y)
    return ds


def main():
    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    model = TinyGPT2().to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ZeRO-1: shard optimizer states across data-parallel ranks.
    # Parameters and gradients are still replicated on each rank.
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Optional mixed precision (BF16 on modern GPUs)
    use_amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 path does not need scaling

    ds = build_fake_data()
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)

    ddp_model.train()
    max_steps = 100

    for step, (inp, tgt) in enumerate(loader):
        if step >= max_steps:
            break
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = ddp_model(inp)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                tgt[:, :-1].contiguous().view(-1),
            )

        loss.backward()
        optimizer.step()

        if step % 10 == 0 and rank == 0:
            print(f"step={step:03d} loss={loss.item():.4f}")

    # Reconstruct full optimizer state on rank 0 for checkpointing.
    if rank == 0:
        optimizer.consolidate_state_dict(to=0)

    dist.barrier()
    if rank == 0:
        ckpt = {
            "model": ddp_model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "world_size": world_size,
        }
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(ckpt, "checkpoints/lesson04_zero1.pt")
        print("Saved checkpoints/lesson04_zero1.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
