#!/usr/bin/env python3
"""
Lesson 22: GQA + KV cache quantization tradeoff simulator.

This lesson models decode-time attention with:
- M query heads
- K KV heads (K <= M, grouped-query attention)
- Optional KV quantization for the cold cache region
- Residual FP16 window for recent tokens

It reports:
- Memory footprint and estimated KV bandwidth reduction
- Output drift vs full MHA FP32 baseline
- Approximate per-token latency proxy

The code is intentionally compact but uses real tensor flows for attention output.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


@dataclass
class Config:
    seq_len: int = 4096
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    decode_steps: int = 128
    residual_window: int = 256
    quant_bits: int = 8  # 8 or 4
    seed: int = 7


@dataclass
class Metrics:
    baseline_kv_bytes: float
    optimized_kv_bytes: float
    memory_reduction_pct: float
    bytes_per_token_baseline: float
    bytes_per_token_optimized: float
    kv_bandwidth_reduction_pct: float
    latency_proxy_speedup: float
    output_mse: float
    output_max_abs: float
    output_cos_sim: float


def bytes_per_scalar(bits: int) -> float:
    return bits / 8.0


def quantize_symmetric(x: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    assert bits in (8, 4)
    qmax = (2 ** (bits - 1)) - 1
    scale = np.max(np.abs(x), axis=0, keepdims=True)
    scale = np.maximum(scale / qmax, 1e-8)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q.astype(np.int8), scale.astype(np.float32)


def dequantize_symmetric(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def group_map(q_heads: int, kv_heads: int) -> np.ndarray:
    assert q_heads % kv_heads == 0
    per_group = q_heads // kv_heads
    idx = np.arange(q_heads) // per_group
    return idx.astype(np.int32)


def build_synthetic_kv(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    # Create latent grouped KV first, then expand to per-query-head KV with small jitter.
    # This makes the baseline closer to a realistic GQA-compatible distribution.
    k_group = rng.standard_normal((cfg.seq_len, cfg.kv_heads, cfg.head_dim), dtype=np.float32)
    v_group = rng.standard_normal((cfg.seq_len, cfg.kv_heads, cfg.head_dim), dtype=np.float32)

    drift = np.linspace(0.0, 0.05, cfg.seq_len, dtype=np.float32)[:, None, None]
    k_group += drift
    v_group -= 0.5 * drift

    mapping = group_map(cfg.q_heads, cfg.kv_heads)
    noise_scale = 0.03
    k_mha = np.empty((cfg.seq_len, cfg.q_heads, cfg.head_dim), dtype=np.float32)
    v_mha = np.empty((cfg.seq_len, cfg.q_heads, cfg.head_dim), dtype=np.float32)
    for h in range(cfg.q_heads):
        g = mapping[h]
        k_mha[:, h, :] = k_group[:, g, :] + noise_scale * rng.standard_normal((cfg.seq_len, cfg.head_dim), dtype=np.float32)
        v_mha[:, h, :] = v_group[:, g, :] + noise_scale * rng.standard_normal((cfg.seq_len, cfg.head_dim), dtype=np.float32)

    return k_mha, v_mha


def mha_to_gqa(k_mha: np.ndarray, v_mha: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    # Collapse M heads -> K KV heads via mean within each query group.
    mapping = group_map(cfg.q_heads, cfg.kv_heads)
    k_gqa = np.zeros((cfg.seq_len, cfg.kv_heads, cfg.head_dim), dtype=np.float32)
    v_gqa = np.zeros((cfg.seq_len, cfg.kv_heads, cfg.head_dim), dtype=np.float32)
    for kvh in range(cfg.kv_heads):
        mask = mapping == kvh
        k_gqa[:, kvh, :] = np.mean(k_mha[:, mask, :], axis=1)
        v_gqa[:, kvh, :] = np.mean(v_mha[:, mask, :], axis=1)
    return k_gqa, v_gqa


def baseline_decode_outputs(k_mha: np.ndarray, v_mha: np.ndarray, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 123)
    q = rng.standard_normal((cfg.decode_steps, cfg.q_heads, cfg.head_dim), dtype=np.float32)

    out = np.empty((cfg.decode_steps, cfg.q_heads, cfg.head_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(cfg.head_dim)
    for t in range(cfg.decode_steps):
        scores = np.einsum("hd,thd->ht", q[t], k_mha) * scale
        probs = softmax(scores, axis=-1)
        out[t] = np.einsum("ht,thd->hd", probs, v_mha)
    return out


def optimized_decode_outputs(k_gqa: np.ndarray, v_gqa: np.ndarray, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 123)
    q = rng.standard_normal((cfg.decode_steps, cfg.q_heads, cfg.head_dim), dtype=np.float32)

    cold = max(cfg.seq_len - cfg.residual_window, 0)
    k_cold = k_gqa[:cold]
    v_cold = v_gqa[:cold]
    k_hot = k_gqa[cold:]
    v_hot = v_gqa[cold:]

    qk_cold, sk_cold = quantize_symmetric(k_cold, bits=cfg.quant_bits)
    qv_cold, sv_cold = quantize_symmetric(v_cold, bits=cfg.quant_bits)

    out = np.empty((cfg.decode_steps, cfg.q_heads, cfg.head_dim), dtype=np.float32)
    gmap = group_map(cfg.q_heads, cfg.kv_heads)
    scale = 1.0 / math.sqrt(cfg.head_dim)

    # Dequantized views (in real systems this should be fused in the kernel)
    k_cold_dq = dequantize_symmetric(qk_cold, sk_cold)
    v_cold_dq = dequantize_symmetric(qv_cold, sv_cold)
    k_mix = np.concatenate([k_cold_dq, k_hot], axis=0)
    v_mix = np.concatenate([v_cold_dq, v_hot], axis=0)

    for t in range(cfg.decode_steps):
        for m in range(cfg.q_heads):
            g = gmap[m]
            scores = np.einsum("d,td->t", q[t, m], k_mix[:, g, :]) * scale
            probs = softmax(scores, axis=-1)
            out[t, m] = np.einsum("t,td->d", probs, v_mix[:, g, :])
    return out


def estimate_memory_and_bw(cfg: Config) -> Dict[str, float]:
    fp16_bytes = 2.0
    cold = max(cfg.seq_len - cfg.residual_window, 0)
    hot = cfg.seq_len - cold

    # Baseline MHA FP16 KV cache: K and V, both [T, M, D]
    baseline = 2.0 * cfg.seq_len * cfg.q_heads * cfg.head_dim * fp16_bytes

    # Optimized GQA + mixed precision KV
    qbytes = bytes_per_scalar(cfg.quant_bits)
    # Cold zone: quantized K/V + fp32 scale metadata per [1,K,D] tensor for K and V
    cold_data = 2.0 * cold * cfg.kv_heads * cfg.head_dim * qbytes
    cold_scales = 2.0 * cfg.kv_heads * cfg.head_dim * 4.0
    hot_data = 2.0 * hot * cfg.kv_heads * cfg.head_dim * fp16_bytes
    optimized = cold_data + cold_scales + hot_data

    # Decode bandwidth proxy: bytes read per token (single step, full context)
    bpt_baseline = 2.0 * cfg.seq_len * cfg.q_heads * cfg.head_dim * fp16_bytes
    bpt_optimized = (
        2.0 * cold * cfg.kv_heads * cfg.head_dim * qbytes
        + 2.0 * hot * cfg.kv_heads * cfg.head_dim * fp16_bytes
    )

    return {
        "baseline_kv_bytes": baseline,
        "optimized_kv_bytes": optimized,
        "memory_reduction_pct": 100.0 * (1.0 - optimized / baseline),
        "bytes_per_token_baseline": bpt_baseline,
        "bytes_per_token_optimized": bpt_optimized,
        "kv_bandwidth_reduction_pct": 100.0 * (1.0 - bpt_optimized / bpt_baseline),
    }


def collect_metrics(cfg: Config) -> Metrics:
    k_mha, v_mha = build_synthetic_kv(cfg)
    k_gqa, v_gqa = mha_to_gqa(k_mha, v_mha, cfg)

    t0 = time.perf_counter()
    y_ref = baseline_decode_outputs(k_mha, v_mha, cfg)
    t1 = time.perf_counter()
    y_opt = optimized_decode_outputs(k_gqa, v_gqa, cfg)
    t2 = time.perf_counter()

    mem = estimate_memory_and_bw(cfg)

    mse = float(np.mean((y_ref - y_opt) ** 2))
    max_abs = float(np.max(np.abs(y_ref - y_opt)))
    cos = float(np.sum(y_ref * y_opt) / (np.linalg.norm(y_ref) * np.linalg.norm(y_opt) + 1e-12))

    # Timing proxy is noisy on CPU; blend with bandwidth model for stable teaching signal.
    wall_speedup = (t1 - t0) / max(t2 - t1, 1e-9)
    bw_speedup = mem["bytes_per_token_baseline"] / max(mem["bytes_per_token_optimized"], 1e-9)
    latency_proxy_speedup = 0.2 * wall_speedup + 0.8 * bw_speedup

    return Metrics(
        baseline_kv_bytes=mem["baseline_kv_bytes"],
        optimized_kv_bytes=mem["optimized_kv_bytes"],
        memory_reduction_pct=mem["memory_reduction_pct"],
        bytes_per_token_baseline=mem["bytes_per_token_baseline"],
        bytes_per_token_optimized=mem["bytes_per_token_optimized"],
        kv_bandwidth_reduction_pct=mem["kv_bandwidth_reduction_pct"],
        latency_proxy_speedup=latency_proxy_speedup,
        output_mse=mse,
        output_max_abs=max_abs,
        output_cos_sim=cos,
    )


def human_bytes(x: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Lesson 22: GQA + KV quantization tradeoff simulator")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--residual-window", type=int, default=256)
    p.add_argument("--quant-bits", type=int, default=8, choices=[8, 4])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    if a.q_heads % a.kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads for grouped mapping.")
    if a.residual_window > a.seq_len:
        raise ValueError("residual_window must be <= seq_len")

    return Config(
        seq_len=a.seq_len,
        q_heads=a.q_heads,
        kv_heads=a.kv_heads,
        head_dim=a.head_dim,
        decode_steps=a.decode_steps,
        residual_window=a.residual_window,
        quant_bits=a.quant_bits,
        seed=a.seed,
    )


def main() -> None:
    cfg = parse_args()
    m = collect_metrics(cfg)

    print("=== Lesson 22: GQA + KV Quantization Report ===")
    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print("\nMemory + Bandwidth:")
    print(f"  baseline_kv: {human_bytes(m.baseline_kv_bytes)}")
    print(f"  optimized_kv: {human_bytes(m.optimized_kv_bytes)}")
    print(f"  memory_reduction_pct: {m.memory_reduction_pct:.2f}%")
    print(f"  bytes_per_token_baseline: {human_bytes(m.bytes_per_token_baseline)}")
    print(f"  bytes_per_token_optimized: {human_bytes(m.bytes_per_token_optimized)}")
    print(f"  kv_bandwidth_reduction_pct: {m.kv_bandwidth_reduction_pct:.2f}%")
    print(f"  latency_proxy_speedup: {m.latency_proxy_speedup:.3f}x")

    print("\nQuality Drift vs FP32 MHA baseline:")
    print(f"  output_mse: {m.output_mse:.6e}")
    print(f"  output_max_abs: {m.output_max_abs:.6e}")
    print(f"  output_cos_sim: {m.output_cos_sim:.6f}")


if __name__ == "__main__":
    main()
