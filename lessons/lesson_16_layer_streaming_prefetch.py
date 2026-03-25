#!/usr/bin/env python3
"""
Lesson 16: Layer Streaming with Asynchronous Prefetch (single-GPU inference)

Goal
----
Run models larger than GPU memory by keeping only one/few layers on device,
streaming weights from CPU RAM, and overlapping H2D copies with compute.

This is a teaching script (not a production engine), but the overlap pattern
is directly transferable to real inference runtimes.

Usage
-----
python lessons/lesson_16_layer_streaming_prefetch.py \
  --layers 24 --hidden 4096 --tokens 512 --dtype fp16 --warmup 2 --steps 10
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class BenchResult:
    mode: str
    ms_per_step: float
    toks_per_s: float


class LayerStreamer:
    def __init__(self, layers: int, hidden: int, dtype: torch.dtype, device: str) -> None:
        self.layers = layers
        self.hidden = hidden
        self.dtype = dtype
        self.device = torch.device(device)

        # Simulate layer weights on CPU RAM (pin for faster async DMA)
        self.w_cpu = [
            torch.randn(hidden, hidden, dtype=dtype, device="cpu").pin_memory()
            for _ in range(layers)
        ]

    @torch.inference_mode()
    def forward_no_overlap(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential: copy layer i then compute layer i (no overlap)."""
        for i in range(self.layers):
            w_dev = self.w_cpu[i].to(self.device, non_blocking=True)
            x = torch.relu(x @ w_dev)
        return x

    @torch.inference_mode()
    def forward_overlap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Double-buffered overlap:
        - copy next layer on prefetch stream
        - compute current layer on default stream
        """
        if self.layers == 0:
            return x

        prefetch_stream = torch.cuda.Stream(device=self.device)

        # Buffer A/B to avoid realloc churn
        w_buf = [
            torch.empty(self.hidden, self.hidden, dtype=self.dtype, device=self.device),
            torch.empty(self.hidden, self.hidden, dtype=self.dtype, device=self.device),
        ]

        # Prefetch layer 0
        with torch.cuda.stream(prefetch_stream):
            w_buf[0].copy_(self.w_cpu[0], non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)

        for i in range(self.layers):
            cur = i % 2
            nxt = (i + 1) % 2

            # Launch prefetch for layer i+1 while computing layer i
            if i + 1 < self.layers:
                with torch.cuda.stream(prefetch_stream):
                    w_buf[nxt].copy_(self.w_cpu[i + 1], non_blocking=True)

            # Compute with current layer
            x = torch.relu(x @ w_buf[cur])

            # Ensure next buffer is ready before next iteration
            if i + 1 < self.layers:
                torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)

        return x


def bench(
    streamer: LayerStreamer,
    mode: str,
    batch: int,
    tokens: int,
    hidden: int,
    warmup: int,
    steps: int,
    dtype: torch.dtype,
    device: str,
) -> BenchResult:
    x = torch.randn(batch * tokens, hidden, dtype=dtype, device=device)

    fn = streamer.forward_no_overlap if mode == "no-overlap" else streamer.forward_overlap

    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = fn(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ms_per_step = dt * 1000 / steps
    toks_per_s = (batch * tokens * steps) / dt
    return BenchResult(mode=mode, ms_per_step=ms_per_step, toks_per_s=toks_per_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this lesson benchmark.")

    dtype = DTYPE_MAP[args.dtype]
    device = "cuda"

    streamer = LayerStreamer(
        layers=args.layers,
        hidden=args.hidden,
        dtype=dtype,
        device=device,
    )

    r0 = bench(streamer, "no-overlap", args.batch, args.tokens, args.hidden, args.warmup, args.steps, dtype, device)
    r1 = bench(streamer, "overlap", args.batch, args.tokens, args.hidden, args.warmup, args.steps, dtype, device)

    speedup = r0.ms_per_step / r1.ms_per_step

    print("=== Lesson 16: Layer Streaming Prefetch Benchmark ===")
    print(f"layers={args.layers} hidden={args.hidden} tokens={args.tokens} batch={args.batch} dtype={args.dtype}")
    print(f"no-overlap : {r0.ms_per_step:8.2f} ms/step | {r0.toks_per_s:,.0f} tok/s")
    print(f"overlap    : {r1.ms_per_step:8.2f} ms/step | {r1.toks_per_s:,.0f} tok/s")
    print(f"speedup    : {speedup:8.3f}x")


if __name__ == "__main__":
    main()
