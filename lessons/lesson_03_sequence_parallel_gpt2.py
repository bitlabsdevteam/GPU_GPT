"""
Lesson 03: Sequence Parallelism with GPT-2 (teaching example)

This file demonstrates the core Sequence Parallelism idea:
- Split tokens across GPUs (sequence dimension) instead of replicating all tokens on each GPU.
- Keep compute local for token-wise ops (LayerNorm, dropout, residual adds).
- Communicate only when an operation needs full-sequence context.

Run (single process demo):
    python lesson_03_sequence_parallel_gpt2.py

Run (distributed, 2+ GPUs):
    torchrun --nproc_per_node=2 lesson_03_sequence_parallel_gpt2.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistEnv:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def init_dist() -> DistEnv:
    """Initialize torch.distributed if launched with torchrun."""
    if "RANK" not in os.environ:
        return DistEnv(enabled=False, rank=0, world_size=1, local_rank=0)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return DistEnv(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def split_sequence(x: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Shard [B, S, H] across sequence dimension S."""
    # Chunk as evenly as possible along sequence axis.
    chunks = torch.chunk(x, world_size, dim=1)
    return chunks[rank].contiguous()


def gather_sequence(local_x: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather local [B, S_local, H] back to [B, S, H]."""
    if world_size == 1:
        return local_x

    gathered = [torch.empty_like(local_x) for _ in range(world_size)]
    dist.all_gather(gathered, local_x)
    return torch.cat(gathered, dim=1)


class GPT2BlockToy(nn.Module):
    """
    Tiny GPT-2-like block (LayerNorm -> MLP -> residual).

    We intentionally keep this simple so the sequence-parallel mechanics stand out.
    """

    def __init__(self, hidden_size: int, mlp_factor: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_factor),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_factor, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token-wise operations: good fit for sequence-parallel local compute.
        h = self.ln(x)
        h = self.mlp(h)
        return x + h


def run_demo() -> None:
    env = init_dist()
    device = torch.device("cuda", env.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(7 + env.rank)

    # Pretend this is a GPT-2 hidden state tensor.
    # [batch, sequence, hidden]
    B, S, H = 2, 16, 64
    x_global = torch.randn(B, S, H, device=device)

    # 1) Sequence-parallel sharding across ranks.
    x_local = split_sequence(x_global, env.world_size, env.rank)

    # 2) Local compute on each rank.
    block = GPT2BlockToy(hidden_size=H).to(device)
    y_local = block(x_local)

    # 3) If later stage needs the full sequence, gather.
    y_global = gather_sequence(y_local, env.world_size)

    if env.rank == 0:
        print(f"Input shape (global):  {tuple(x_global.shape)}")
        print(f"Shard shape (per rank): {tuple(x_local.shape)}")
        print(f"Output shape (global): {tuple(y_global.shape)}")

    if env.enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()
