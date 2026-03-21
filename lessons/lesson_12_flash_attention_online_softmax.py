"""
Lesson 12: FlashAttention-style blockwise attention with online softmax.

This file demonstrates the core streaming recurrence used in memory-efficient
attention kernels. It is intentionally written in plain PyTorch for clarity.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import torch


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Reference attention using full score matrix materialization.

    Shapes:
      q, k, v: [B, H, T, D]
      returns: [B, H, T, D]
    """
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)

    if causal:
        t_q, t_k = q.size(-2), k.size(-2)
        mask = torch.triu(
            torch.ones(t_q, t_k, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def flash_attention_blockwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 128,
) -> torch.Tensor:
    """Blockwise streaming attention with online softmax.

    This mirrors the high-level algorithm used by FlashAttention kernels:
      - process Q in blocks of block_m rows
      - stream over K/V in blocks of block_n columns
      - maintain online softmax stats (m, l, acc)

    Shapes:
      q, k, v: [B, H, T, D]
      returns: [B, H, T, D]
    """
    assert q.shape == k.shape == v.shape
    bsz, n_heads, t, d = q.shape
    scale = d ** -0.5

    out = torch.empty_like(q)

    for qs in range(0, t, block_m):
        qe = min(qs + block_m, t)
        q_blk = q[:, :, qs:qe, :]  # [B, H, M, D]
        m = torch.full((bsz, n_heads, qe - qs), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((bsz, n_heads, qe - qs), device=q.device, dtype=torch.float32)
        acc = torch.zeros((bsz, n_heads, qe - qs, d), device=q.device, dtype=torch.float32)

        q_idx = torch.arange(qs, qe, device=q.device).view(1, 1, -1, 1)

        for ks in range(0, t, block_n):
            ke = min(ks + block_n, t)
            k_blk = k[:, :, ks:ke, :]  # [B, H, N, D]
            v_blk = v[:, :, ks:ke, :]  # [B, H, N, D]

            scores = torch.matmul(q_blk.float(), k_blk.float().transpose(-2, -1)) * scale

            if causal:
                k_idx = torch.arange(ks, ke, device=q.device).view(1, 1, 1, -1)
                local_mask = k_idx > q_idx
                scores = scores.masked_fill(local_mask, float("-inf"))

            m_blk = torch.amax(scores, dim=-1)  # [B, H, M]
            m_new = torch.maximum(m, m_blk)

            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new.unsqueeze(-1))

            l = alpha * l + torch.sum(p, dim=-1)
            acc = alpha.unsqueeze(-1) * acc + torch.matmul(p, v_blk.float())
            m = m_new

        out[:, :, qs:qe, :] = (acc / l.unsqueeze(-1)).to(q.dtype)

    return out


@torch.no_grad()
def benchmark(
    seq_lens: Iterable[int],
    batch_size: int,
    heads: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    causal: bool,
    block_m: int,
    block_n: int,
    warmup: int,
    iters: int,
) -> None:
    print(f"device={device} dtype={dtype} batch={batch_size} heads={heads} d={head_dim}")
    for t in seq_lens:
        q = torch.randn(batch_size, heads, t, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        ref = naive_attention(q, k, v, causal=causal)
        out = flash_attention_blockwise(q, k, v, causal=causal, block_m=block_m, block_n=block_n)

        max_abs = (ref - out).abs().max().item()
        ok = torch.allclose(ref, out, atol=3e-2, rtol=3e-2)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = naive_attention(q, k, v, causal=causal)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = naive_attention(q, k, v, causal=causal)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) * 1000 / iters

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = flash_attention_blockwise(q, k, v, causal=causal, block_m=block_m, block_n=block_n)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = flash_attention_blockwise(q, k, v, causal=causal, block_m=block_m, block_n=block_n)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        block_ms = (time.perf_counter() - t0) * 1000 / iters

        print(
            f"T={t:4d} | max_abs={max_abs:.4e} | allclose={ok} | "
            f"naive={naive_ms:8.2f} ms | blockwise={block_ms:8.2f} ms"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seq-lens", type=str, default="256,512,1024")
    p.add_argument("--non-causal", action="store_true")
    p.add_argument("--block-m", type=int, default=64)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    return p.parse_args()


def to_dtype(s: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[s]


if __name__ == "__main__":
    args = parse_args()
    dtype = to_dtype(args.dtype)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]

    benchmark(
        seq_lens=seq_lens,
        batch_size=args.batch_size,
        heads=args.heads,
        head_dim=args.head_dim,
        device=args.device,
        dtype=dtype,
        causal=not args.non_causal,
        block_m=args.block_m,
        block_n=args.block_n,
        warmup=args.warmup,
        iters=args.iters,
    )
