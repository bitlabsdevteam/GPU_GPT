#!/usr/bin/env python3
"""
Lesson 14: Activation Checkpointing for GPT Blocks

Goal:
- Quantify memory-vs-throughput tradeoff from activation checkpointing.
- Compare eager vs selective recompute on a GPT-like stack.
- Provide production-style metrics (peak memory, step time, tokens/s).

Usage:
  python lessons/lesson_14_activation_checkpointing_selective_recompute.py \
      --device cuda --dtype bf16 --batch-size 8 --seq-len 1024 \
      --d-model 768 --n-heads 12 --n-layers 12 --iters 30 --warmup-iters 10

CPU fallback:
  python lessons/lesson_14_activation_checkpointing_selective_recompute.py --device cpu --iters 10
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(out))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(mult * d_model)
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.out(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, use_ckpt: bool, ckpt_every: int) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)

        for i, block in enumerate(self.blocks):
            if use_ckpt and (i % ckpt_every == 0):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        return self.head(x)


@dataclass
class BenchResult:
    mode: str
    ms_per_iter: float
    tokens_per_s: float
    peak_mem_mb: float


def autocast_context(device: str, dtype: str):
    if device != "cuda":
        return contextlib.nullcontext()
    if dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def run_benchmark(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    iters: int,
    warmup: int,
    device: str,
    dtype: str,
    use_ckpt: bool,
    ckpt_every: int,
) -> BenchResult:
    model.train()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            logits = model(input_ids, use_ckpt=use_ckpt, ckpt_every=ckpt_every)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            logits = model(input_ids, use_ckpt=use_ckpt, ckpt_every=ckpt_every)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_iter = (elapsed / iters) * 1000.0
    tokens_per_iter = input_ids.numel()
    toks_per_s = tokens_per_iter / (elapsed / iters)

    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem_mb = float("nan")

    return BenchResult(
        mode=f"checkpoint_every_{ckpt_every}" if use_ckpt else "eager_no_checkpoint",
        ms_per_iter=ms_per_iter,
        tokens_per_s=toks_per_s,
        peak_mem_mb=peak_mem_mb,
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Activation checkpointing tradeoff benchmark")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt-every", type=int, default=1, help="Checkpoint every N-th block")
    return p


def main() -> None:
    args = make_parser().parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TinyGPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    ).to(args.device)

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=args.device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    eager = run_benchmark(
        model,
        optimizer,
        input_ids,
        targets,
        iters=args.iters,
        warmup=args.warmup_iters,
        device=args.device,
        dtype=args.dtype,
        use_ckpt=False,
        ckpt_every=args.ckpt_every,
    )

    ckpt = run_benchmark(
        model,
        optimizer,
        input_ids,
        targets,
        iters=args.iters,
        warmup=args.warmup_iters,
        device=args.device,
        dtype=args.dtype,
        use_ckpt=True,
        ckpt_every=max(1, args.ckpt_every),
    )

    print("\n=== Activation Checkpointing Benchmark ===")
    print(f"device={args.device} dtype={args.dtype} batch={args.batch_size} seq={args.seq_len} layers={args.n_layers}")
    print(f"eager: ms/iter={eager.ms_per_iter:.2f}, toks/s={eager.tokens_per_s:,.0f}, peak_mem_mb={eager.peak_mem_mb:.1f}")
    print(f"ckpt : ms/iter={ckpt.ms_per_iter:.2f}, toks/s={ckpt.tokens_per_s:,.0f}, peak_mem_mb={ckpt.peak_mem_mb:.1f}")

    if math.isfinite(eager.peak_mem_mb) and math.isfinite(ckpt.peak_mem_mb):
        mem_saved = eager.peak_mem_mb - ckpt.peak_mem_mb
        mem_saved_pct = (mem_saved / eager.peak_mem_mb) * 100.0 if eager.peak_mem_mb > 0 else float("nan")
        print(f"memory_saved_mb={mem_saved:.1f} ({mem_saved_pct:.1f}%)")

    speed_ratio = eager.ms_per_iter / ckpt.ms_per_iter if ckpt.ms_per_iter > 0 else float("nan")
    print(f"throughput_ratio_ckpt_vs_eager={speed_ratio:.3f}x (higher means checkpoint mode is faster)")


if __name__ == "__main__":
    main()
