#!/usr/bin/env python3
"""
Lesson 13: CUDA Graphs for low-latency, stable-shape GPT training/inference steps.

What this script demonstrates:
1) Eager training step (forward + backward + optimizer step)
2) CUDA Graph captured training step with static buffers
3) Timing comparison over many iterations

Why this matters for LLM systems:
- Python/kernel launch overhead can dominate at small batch sizes or decode-time loops.
- CUDA Graphs reduce per-iteration launch overhead by replaying a captured graph.
- Works best when tensor shapes and control flow are static.

Run (GPU):
  python lessons/lesson_13_cuda_graphs_training_step.py --device cuda --dtype bf16 --iters 200

Run (CPU fallback for structure only):
  python lessons/lesson_13_cuda_graphs_training_step.py --device cpu --iters 50
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = mlp_ratio * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def _causal_attention(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        b, t, c = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, T, D]
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,T,T]

        # causal mask
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,T,D]

        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self._causal_attention(self.ln1(x)))
        x = x + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh")))
        return x


class TinyGPTLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TinyGPTBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(input_ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits


@dataclass
class StepArtifacts:
    static_input_ids: torch.Tensor
    static_targets: torch.Tensor
    loss_buf: torch.Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_step(model: nn.Module, optimizer: torch.optim.Optimizer, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    return loss.detach()


def warmup(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
) -> None:
    for _ in range(steps):
        _ = training_step(model, optimizer, input_ids, targets)


def build_static_buffers(batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    static_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    static_targets = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    return static_input_ids, static_targets


def capture_training_graph(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    artifacts: StepArtifacts,
) -> torch.cuda.CUDAGraph:
    g = torch.cuda.CUDAGraph()

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        logits = model(artifacts.static_input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), artifacts.static_targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        artifacts.loss_buf.copy_(loss.detach())

    return g


def random_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def bench_eager(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iters: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(iters):
        x, y = random_batch(batch_size, seq_len, vocab_size, device)
        last_loss = float(training_step(model, optimizer, x, y))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / iters, last_loss


def bench_cuda_graph(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iters: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    static_input_ids, static_targets = build_static_buffers(batch_size, seq_len, device)
    loss_buf = torch.zeros((), device=device, dtype=torch.float32)
    artifacts = StepArtifacts(static_input_ids, static_targets, loss_buf)

    # Warmup on a side stream to initialize CUDA kernels/allocations before capture.
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        xw, yw = random_batch(batch_size, seq_len, vocab_size, device)
        warmup(model, optimizer, xw, yw, steps=8)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = capture_training_graph(model, optimizer, artifacts)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x, y = random_batch(batch_size, seq_len, vocab_size, device)
        artifacts.static_input_ids.copy_(x)
        artifacts.static_targets.copy_(y)
        graph.replay()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return dt / iters, float(artifacts.loss_buf.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lesson 13: CUDA Graphs training-step benchmark")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)

    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_str]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable. Falling back to CPU.")
        requested_device = torch.device("cpu")

    dtype = resolve_dtype(args.dtype)
    if requested_device.type == "cpu" and dtype != torch.float32:
        print("[warn] Non-fp32 dtype on CPU is not generally beneficial; using fp32.")
        dtype = torch.float32

    model = TinyGPTLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    ).to(requested_device)

    # Keep master params in fp32 for optimizer stability.
    # Use autocast for compute dtype when on CUDA.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initial warmup for eager baseline.
    xw, yw = random_batch(args.batch_size, args.seq_len, args.vocab_size, requested_device)
    warmup(model, optimizer, xw, yw, steps=args.warmup_iters)

    if requested_device.type == "cuda":
        eager_ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        eager_ctx = torch.autocast(device_type="cpu", enabled=False)

    # Wrap model forward in autocast by monkey-patching training_step usage via context.
    # Keep benchmark logic simple and explicit.
    global training_step  # noqa: PLW0603
    base_training_step = training_step

    def training_step_autocast(m: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        with eager_ctx:
            return base_training_step(m, opt, inp, tgt)

    training_step = training_step_autocast

    eager_ms, eager_loss = bench_eager(
        model,
        optimizer,
        args.iters,
        args.batch_size,
        args.seq_len,
        args.vocab_size,
        requested_device,
    )

    print("=== Lesson 13: CUDA Graphs Benchmark ===")
    print(f"device={requested_device.type} dtype={dtype} batch={args.batch_size} seq={args.seq_len}")
    print(f"eager:      {eager_ms * 1e3:.3f} ms/iter | last_loss={eager_loss:.6f}")

    if requested_device.type != "cuda":
        print("cuda_graph: skipped (requires CUDA)")
        return

    # For fair comparison, build a fresh model for graph path.
    model_g = TinyGPTLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    ).to(requested_device)
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # warmup under autocast
    xw2, yw2 = random_batch(args.batch_size, args.seq_len, args.vocab_size, requested_device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        warmup(model_g, optimizer_g, xw2, yw2, steps=args.warmup_iters)

    # Keep graph capture under autocast context for compute dtype.
    orig_capture = capture_training_graph

    def capture_with_autocast(model: nn.Module, optimizer: torch.optim.Optimizer, artifacts: StepArtifacts) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(artifacts.static_input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), artifacts.static_targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            artifacts.loss_buf.copy_(loss.detach().float())
        return g

    globals()["capture_training_graph"] = capture_with_autocast

    graph_ms, graph_loss = bench_cuda_graph(
        model_g,
        optimizer_g,
        args.iters,
        args.batch_size,
        args.seq_len,
        args.vocab_size,
        requested_device,
    )

    speedup = eager_ms / graph_ms if graph_ms > 0 else float("inf")
    print(f"cuda_graph: {graph_ms * 1e3:.3f} ms/iter | last_loss={graph_loss:.6f}")
    print(f"speedup:    {speedup:.3f}x")

    # restore function references for cleanliness
    globals()["capture_training_graph"] = orig_capture
    training_step = base_training_step


if __name__ == "__main__":
    main()
