#!/usr/bin/env python3
"""
Lesson 21: INT8 KV-cache quantization with residual FP16 window.

Why this matters:
- Decode is often memory-bandwidth bound.
- KV cache dominates HBM usage at long context.
- Quantizing cold KV pages reduces bytes/token and improves effective throughput.

This script simulates a production pattern:
1) Keep the newest tokens in FP16 (residual window) for quality-critical recency.
2) Quantize older KV blocks to INT8 with per-head/per-channel scales.
3) Dequantize on read and compare quality/latency/memory against FP16 baseline.

Run:
  python3 lessons/lesson_21_kv_cache_int8_quantization.py --help
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class QuantStats:
    mse: float
    max_abs: float
    cos_sim: float


def build_synthetic_kv(seq_len: int, n_heads: int, head_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = rng.normal(0, 0.8, size=(seq_len, n_heads, head_dim)).astype(np.float32)
    v = rng.normal(0, 0.8, size=(seq_len, n_heads, head_dim)).astype(np.float32)

    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)[:, None, None]
    drift = 0.15 * np.sin(6.0 * math.pi * t)
    k += drift
    v += 0.7 * drift
    return k, v


def quantize_int8_per_head_channel(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x: [T, H, D]
    max_abs = np.max(np.abs(x), axis=0, keepdims=True) + 1e-8  # [1,H,D]
    scale = max_abs / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale


def quant_stats(x_ref: np.ndarray, x_hat: np.ndarray) -> QuantStats:
    diff = x_ref - x_hat
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    ref_flat = x_ref.reshape(-1)
    hat_flat = x_hat.reshape(-1)
    denom = np.linalg.norm(ref_flat) * np.linalg.norm(hat_flat) + 1e-12
    cos = float(np.dot(ref_flat, hat_flat) / denom)
    return QuantStats(mse=mse, max_abs=max_abs, cos_sim=cos)


def attention_context(q_t: np.ndarray, k_ctx: np.ndarray, v_ctx: np.ndarray, temp: float = 1.0) -> np.ndarray:
    # q_t: [H,D], k_ctx/v_ctx: [T,H,D]
    logits = np.einsum("hd,thd->th", q_t, k_ctx) / math.sqrt(q_t.shape[-1])
    logits = logits / max(temp, 1e-8)
    logits = logits - np.max(logits, axis=0, keepdims=True)
    probs = np.exp(logits)
    probs = probs / (np.sum(probs, axis=0, keepdims=True) + 1e-12)
    ctx = np.einsum("th,thd->hd", probs, v_ctx)
    return ctx


def benchmark_decode(
    q_seq: np.ndarray,
    k_ref: np.ndarray,
    v_ref: np.ndarray,
    residual_window: int,
) -> dict:
    t_total = k_ref.shape[0]
    cut = max(0, t_total - residual_window)

    k_old, v_old = k_ref[:cut], v_ref[:cut]
    k_new, v_new = k_ref[cut:], v_ref[cut:]

    qk, sk = quantize_int8_per_head_channel(k_old) if cut > 0 else (None, None)
    qv, sv = quantize_int8_per_head_channel(v_old) if cut > 0 else (None, None)

    # Baseline FP16 memory footprint (bytes)
    fp16_bytes = int((k_ref.size + v_ref.size) * 2)

    # Quantized footprint: int8 for old + fp16 for residual + scales in fp32
    int8_old_bytes = int((k_old.size + v_old.size) * 1)
    fp16_new_bytes = int((k_new.size + v_new.size) * 2)
    scale_bytes = 0 if cut == 0 else int((sk.size + sv.size) * 4)
    quant_bytes = int8_old_bytes + fp16_new_bytes + scale_bytes

    # Decode benchmark: compare FP32 ref vs quantized reconstruction
    n_steps = q_seq.shape[0]
    ref_out = []
    q_out = []

    t0 = time.perf_counter()
    for i in range(n_steps):
        ref_out.append(attention_context(q_seq[i], k_ref, v_ref))
    ref_ms = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    if cut > 0:
        k_old_hat = dequantize_int8(qk, sk)
        v_old_hat = dequantize_int8(qv, sv)
        k_mix = np.concatenate([k_old_hat, k_new], axis=0)
        v_mix = np.concatenate([v_old_hat, v_new], axis=0)
    else:
        k_mix, v_mix = k_ref, v_ref

    for i in range(n_steps):
        q_out.append(attention_context(q_seq[i], k_mix, v_mix))
    quant_ms = (time.perf_counter() - t1) * 1e3

    ref_out = np.stack(ref_out)
    q_out = np.stack(q_out)

    out_stats = quant_stats(ref_out, q_out)

    return {
        "fp16_bytes": fp16_bytes,
        "quant_bytes": quant_bytes,
        "memory_reduction_pct": 100.0 * (1.0 - quant_bytes / max(fp16_bytes, 1)),
        "ref_decode_ms": ref_ms,
        "quant_decode_ms": quant_ms,
        "speedup": ref_ms / max(quant_ms, 1e-9),
        "output_mse": out_stats.mse,
        "output_max_abs": out_stats.max_abs,
        "output_cos_sim": out_stats.cos_sim,
        "cut_tokens": cut,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lesson 21: INT8 KV cache quantization + residual FP16 window")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--residual-window", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    k, v = build_synthetic_kv(args.seq_len, args.heads, args.head_dim, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    q_seq = rng.normal(0, 1.0, size=(args.decode_steps, args.heads, args.head_dim)).astype(np.float32)

    res = benchmark_decode(q_seq, k, v, args.residual_window)

    print("=== Lesson 21: INT8 KV-cache quantization (residual FP16 window) ===")
    print(f"seq_len={args.seq_len}, heads={args.heads}, head_dim={args.head_dim}, decode_steps={args.decode_steps}")
    print(f"residual_window={args.residual_window}, quantized_tokens={res['cut_tokens']}")
    print("--- Memory ---")
    print(f"fp16_bytes={res['fp16_bytes']:,}")
    print(f"quant_bytes={res['quant_bytes']:,}")
    print(f"memory_reduction_pct={res['memory_reduction_pct']:.2f}%")
    print("--- Decode timing (CPU simulation) ---")
    print(f"ref_decode_ms={res['ref_decode_ms']:.2f}")
    print(f"quant_decode_ms={res['quant_decode_ms']:.2f}")
    print(f"speedup={res['speedup']:.3f}x")
    print("--- Output drift vs FP baseline ---")
    print(f"output_mse={res['output_mse']:.6e}")
    print(f"output_max_abs={res['output_max_abs']:.6e}")
    print(f"output_cos_sim={res['output_cos_sim']:.9f}")

    print("\nNotes:")
    print("1) In production, quantize by pages/chunks and fuse dequant into attention kernels.")
    print("2) Keep recent tokens FP16/BF16 to preserve near-context quality.")
    print("3) Use calibration + A/B eval to set residual window and quant granularity.")


if __name__ == "__main__":
    main()
