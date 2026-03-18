"""
Lesson 09: Context Parallelism for long-context LLMs.

What this demonstrates:
- Sequence sharding across context-parallel ranks
- Exact blockwise attention accumulation via online softmax merge
- A ring-style KV exchange simulation (single-process didactic model)
- Communication-volume estimation vs tensor shapes

Run:
  python GPU_GPT/lessons/lesson_09_context_parallelism_ring_attention.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Config:
    batch: int = 2
    heads: int = 4
    seq_len: int = 1024
    head_dim: int = 64
    cp_ranks: int = 4
    dtype: torch.dtype = torch.float32
    seed: int = 7


def split_context(x: torch.Tensor, cp_ranks: int) -> List[torch.Tensor]:
    """Split tensor [B, H, S, D] into cp_ranks chunks on sequence dimension."""
    return list(torch.chunk(x, cp_ranks, dim=2))


def full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference full attention for correctness checks."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def online_softmax_merge(
    prev_m: torch.Tensor,
    prev_l: torch.Tensor,
    prev_out: torch.Tensor,
    block_scores: torch.Tensor,
    block_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge one score block into running attention using numerically stable online softmax.

    Shapes:
      prev_m, prev_l: [B,H,S_local,1]
      prev_out: [B,H,S_local,D]
      block_scores: [B,H,S_local,S_block]
      block_v: [B,H,S_block,D]
    """
    block_m = block_scores.max(dim=-1, keepdim=True).values
    block_exp = torch.exp(block_scores - block_m)
    block_l = block_exp.sum(dim=-1, keepdim=True)
    block_out = torch.matmul(block_exp, block_v)

    new_m = torch.maximum(prev_m, block_m)
    prev_scale = torch.exp(prev_m - new_m)
    block_scale = torch.exp(block_m - new_m)

    new_l = prev_scale * prev_l + block_scale * block_l
    new_out = prev_scale * prev_out + block_scale * block_out
    return new_m, new_l, new_out


def ring_context_parallel_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp_ranks: int) -> torch.Tensor:
    """
    Simulate ring-style context parallel attention.

    Each rank owns a local Q,K,V sequence chunk. For output of local Q, it iterates over
    all KV chunks in ring order and merges partial attention with online softmax.
    """
    q_chunks = split_context(q, cp_ranks)
    k_chunks = split_context(k, cp_ranks)
    v_chunks = split_context(v, cp_ranks)

    scale = 1.0 / math.sqrt(q.size(-1))
    local_outputs = []

    for r in range(cp_ranks):
        q_local = q_chunks[r]  # [B,H,S_local,D]
        b, h, s_local, d = q_local.shape

        m = torch.full((b, h, s_local, 1), -float("inf"), dtype=q.dtype)
        l = torch.zeros((b, h, s_local, 1), dtype=q.dtype)
        out = torch.zeros((b, h, s_local, d), dtype=q.dtype)

        # Ring order: local first, then neighbors. Exact output is order-invariant with merge.
        for step in range(cp_ranks):
            kv_owner = (r + step) % cp_ranks
            k_block = k_chunks[kv_owner]
            v_block = v_chunks[kv_owner]

            block_scores = torch.matmul(q_local, k_block.transpose(-1, -2)) * scale
            m, l, out = online_softmax_merge(m, l, out, block_scores, v_block)

        local_outputs.append(out / l.clamp_min(1e-9))

    return torch.cat(local_outputs, dim=2)


def estimate_ring_comm_bytes(cfg: Config) -> int:
    """
    Rough all-gather/ring communication estimate for one attention layer.

    Per ring step each rank sends one K and one V block of shape [B,H,S_local,D].
    There are (cp_ranks - 1) exchange steps in a physical ring implementation.
    """
    s_local = cfg.seq_len // cfg.cp_ranks
    elems_per_block = cfg.batch * cfg.heads * s_local * cfg.head_dim
    bytes_per_elem = 4 if cfg.dtype == torch.float32 else 2

    kv_bytes_per_step = 2 * elems_per_block * bytes_per_elem
    total_per_rank = (cfg.cp_ranks - 1) * kv_bytes_per_step
    total_cluster = cfg.cp_ranks * total_per_rank
    return total_cluster


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}"


def main() -> None:
    cfg = Config()
    assert cfg.seq_len % cfg.cp_ranks == 0, "seq_len must divide cp_ranks"

    torch.manual_seed(cfg.seed)

    q = torch.randn(cfg.batch, cfg.heads, cfg.seq_len, cfg.head_dim, dtype=cfg.dtype)
    k = torch.randn(cfg.batch, cfg.heads, cfg.seq_len, cfg.head_dim, dtype=cfg.dtype)
    v = torch.randn(cfg.batch, cfg.heads, cfg.seq_len, cfg.head_dim, dtype=cfg.dtype)

    y_full = full_attention(q, k, v)
    y_ring = ring_context_parallel_attention(q, k, v, cfg.cp_ranks)

    max_abs_err = (y_full - y_ring).abs().max().item()
    mean_abs_err = (y_full - y_ring).abs().mean().item()

    comm_bytes = estimate_ring_comm_bytes(cfg)
    print("=== Lesson 09: Context Parallelism / Ring Attention (didactic) ===")
    print(
        f"shape: B={cfg.batch} H={cfg.heads} S={cfg.seq_len} D={cfg.head_dim} cp_ranks={cfg.cp_ranks}"
    )
    print(f"correctness: max_abs_err={max_abs_err:.6e} mean_abs_err={mean_abs_err:.6e}")
    print(f"estimated ring KV traffic per layer (cluster-wide): {fmt_bytes(comm_bytes)}")

    # Small sweep to show communication trend with context length.
    print("\nComm sweep (holding B,H,D,cp_ranks fixed):")
    for s in [4096, 8192, 16384, 32768]:
        c2 = Config(
            batch=cfg.batch,
            heads=cfg.heads,
            seq_len=s,
            head_dim=cfg.head_dim,
            cp_ranks=cfg.cp_ranks,
            dtype=cfg.dtype,
            seed=cfg.seed,
        )
        print(f"  S={s:5d} -> {fmt_bytes(estimate_ring_comm_bytes(c2))}")


if __name__ == "__main__":
    main()
