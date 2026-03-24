#!/usr/bin/env python3
"""
Lesson 15: Overlapping CPU Offload with GPU Compute (ZeRO-Offload pattern)

Goal:
- Demonstrate why async prefetch/evict pipelines reduce offload overhead.
- Compare a no-overlap baseline vs double-buffered overlap.

Example:
  python lessons/lesson_15_zero_offload_overlap.py \
    --device cuda --num-shards 24 --shard-numel 8000000 --compute-iters 80
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


@dataclass
class Bench:
    mode: str
    ms: float


def simulated_optimizer_compute(t: torch.Tensor, compute_iters: int) -> torch.Tensor:
    # Synthetic compute to emulate optimizer math cost.
    x = t
    for _ in range(compute_iters):
        x = x.mul_(0.99999).add_(1e-6)
    return x


def make_cpu_shards(num_shards: int, shard_numel: int, pin: bool) -> list[torch.Tensor]:
    shards = []
    for _ in range(num_shards):
        cpu = torch.randn(shard_numel, dtype=torch.float32, device="cpu")
        if pin:
            cpu = cpu.pin_memory()
        shards.append(cpu)
    return shards


def baseline_no_overlap(cpu_shards: list[torch.Tensor], device: str, compute_iters: int) -> Bench:
    if device != "cuda":
        start = time.perf_counter()
        for s in cpu_shards:
            x = s.clone()
            simulated_optimizer_compute(x, max(1, compute_iters // 10))
        return Bench("baseline_no_overlap_cpu", (time.perf_counter() - start) * 1000.0)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for s in cpu_shards:
        gpu = s.to("cuda", non_blocking=True)
        simulated_optimizer_compute(gpu, compute_iters)
        _ = gpu.to("cpu", non_blocking=True)

    torch.cuda.synchronize()
    return Bench("baseline_no_overlap", (time.perf_counter() - start) * 1000.0)


def overlapped_double_buffer(cpu_shards: list[torch.Tensor], compute_iters: int) -> Bench:
    assert torch.cuda.is_available()

    prefetch_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    ready_events = [torch.cuda.Event() for _ in range(2)]
    buffers = [None, None]

    n = len(cpu_shards)
    torch.cuda.synchronize()
    start = time.perf_counter()

    # Prime first prefetch
    with torch.cuda.stream(prefetch_stream):
        buffers[0] = cpu_shards[0].to("cuda", non_blocking=True)
        ready_events[0].record(prefetch_stream)

    for i in range(n):
        cur = i % 2
        nxt = (i + 1) % 2

        # Launch next prefetch while current shard computes.
        if i + 1 < n:
            with torch.cuda.stream(prefetch_stream):
                buffers[nxt] = cpu_shards[i + 1].to("cuda", non_blocking=True)
                ready_events[nxt].record(prefetch_stream)

        compute_stream.wait_event(ready_events[cur])
        gpu = buffers[cur]
        simulated_optimizer_compute(gpu, compute_iters)

        # Async evict (simulated writeback)
        _ = gpu.to("cpu", non_blocking=True)

    torch.cuda.synchronize()
    return Bench("overlap_double_buffer", (time.perf_counter() - start) * 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CPU offload overlap pattern")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    p.add_argument("--num-shards", type=int, default=24)
    p.add_argument("--shard-numel", type=int, default=8_000_000)
    p.add_argument("--compute-iters", type=int, default=80)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    pin = args.device == "cuda"
    cpu_shards = make_cpu_shards(args.num_shards, args.shard_numel, pin=pin)

    # Warmup
    for _ in range(args.warmup):
        _ = baseline_no_overlap(cpu_shards[: min(4, len(cpu_shards))], args.device, max(4, args.compute_iters // 4))

    base = baseline_no_overlap(cpu_shards, args.device, args.compute_iters)

    if args.device == "cuda":
        ov = overlapped_double_buffer(cpu_shards, args.compute_iters)
        speedup = base.ms / ov.ms if ov.ms > 0 else float("inf")

        print("\n=== Lesson 15 Benchmark: CPU Offload Overlap ===")
        print(f"Shards: {args.num_shards}, shard_numel: {args.shard_numel:,}, compute_iters: {args.compute_iters}")
        print(f"{base.mode:>24}: {base.ms:9.2f} ms")
        print(f"{ov.mode:>24}: {ov.ms:9.2f} ms")
        print(f"{'speedup':>24}: {speedup:9.3f}x")
    else:
        print("\nCPU-only mode: validated control flow (no true DMA overlap).")
        print(f"{base.mode}: {base.ms:.2f} ms")


if __name__ == "__main__":
    main()
