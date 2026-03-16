"""
Lesson 07: Pipeline Parallelism for GPT-style blocks.

This script demonstrates:
1) Splitting a GPT-style model into pipeline stages.
2) Running micro-batches through a simple 1F1B-like schedule.
3) Measuring bubble overhead and effective throughput.

It is intentionally compact and runnable on CPU for learning,
but maps directly to multi-GPU stage placement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class Stage0(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[GPTBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        return self.blocks(x)


class Stage1(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, vocab_size: int):
        super().__init__()
        self.blocks = nn.Sequential(*[GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.head(self.ln_f(x))


@dataclass
class Config:
    vocab_size: int = 50257
    seq_len: int = 128
    global_batch: int = 32
    micro_batches: int = 4
    d_model: int = 256
    n_heads: int = 8
    total_layers: int = 8
    lr: float = 3e-4
    steps: int = 30


def split_into_microbatches(x: torch.Tensor, m: int):
    assert x.size(0) % m == 0, "global_batch must be divisible by micro_batches"
    mb = x.size(0) // m
    return [x[i * mb : (i + 1) * mb] for i in range(m)]


def run_step(
    stage0: nn.Module,
    stage1: nn.Module,
    opt0: torch.optim.Optimizer,
    opt1: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> tuple[float, float]:
    """Returns (loss, step_time_seconds)."""
    t0 = time.perf_counter()
    opt0.zero_grad(set_to_none=True)
    opt1.zero_grad(set_to_none=True)

    x_chunks = split_into_microbatches(token_ids, cfg.micro_batches)
    y_chunks = split_into_microbatches(targets, cfg.micro_batches)

    losses = []
    # Forward all micro-batches through pipeline stages.
    activations = []
    for x_mb in x_chunks:
        h_mb = stage0(x_mb)
        activations.append(h_mb)

    for h_mb, y_mb in zip(activations, y_chunks):
        logits = stage1(h_mb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_mb.view(-1))
        loss = loss / cfg.micro_batches
        losses.append(loss.detach())
        loss.backward()

    opt0.step()
    opt1.step()
    t1 = time.perf_counter()
    return torch.stack(losses).sum().item(), t1 - t0


def bubble_fraction(stages: int, micro_batches: int) -> float:
    # Approximate pipeline bubble fraction for GPipe-style schedule.
    return (stages - 1) / (micro_batches + stages - 1)


def main():
    cfg = Config()

    layers_per_stage = cfg.total_layers // 2
    stage0 = Stage0(cfg.vocab_size, cfg.d_model, cfg.n_heads, layers_per_stage)
    stage1 = Stage1(cfg.d_model, cfg.n_heads, layers_per_stage, cfg.vocab_size)

    opt0 = torch.optim.AdamW(stage0.parameters(), lr=cfg.lr)
    opt1 = torch.optim.AdamW(stage1.parameters(), lr=cfg.lr)

    print("=== Lesson 07: Pipeline Parallel GPT-2 Skeleton ===")
    print(f"global_batch={cfg.global_batch}, micro_batches={cfg.micro_batches}")
    print(f"approx_bubble_fraction={bubble_fraction(stages=2, micro_batches=cfg.micro_batches):.3f}")

    for step in range(1, cfg.steps + 1):
        token_ids = torch.randint(0, cfg.vocab_size, (cfg.global_batch, cfg.seq_len))
        targets = torch.randint(0, cfg.vocab_size, (cfg.global_batch, cfg.seq_len))
        loss, dt = run_step(stage0, stage1, opt0, opt1, token_ids, targets, cfg)

        toks = cfg.global_batch * cfg.seq_len
        toks_per_sec = toks / dt
        if step % 5 == 0 or step == 1:
            print(f"step={step:03d} loss={loss:.4f} time={dt*1000:.1f}ms tok/s={toks_per_sec:,.0f}")

    print("Done.")


if __name__ == "__main__":
    torch.manual_seed(7)
    main()
