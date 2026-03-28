#!/usr/bin/env python3
"""
Lesson 19: Paged KV allocator simulation under continuous batching.

Focus:
- Fixed-size KV blocks and per-sequence block tables
- Allocation/free churn from arrivals + completions
- Prefix-sharing impact on memory pressure
- Fragmentation and allocator health metrics

This is a systems lesson for LLM serving policy design (no GPU required).
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(order=True)
class Arrival:
    t: int
    req_id: int = field(compare=False)
    prompt_tokens: int = field(compare=False)
    gen_tokens: int = field(compare=False)
    prefix_key: int = field(compare=False)


@dataclass
class Request:
    req_id: int
    t_arrival: int
    prompt_tokens_total: int
    gen_tokens_target: int
    prefix_key: int

    prompt_tokens_done: int = 0
    generated: int = 0
    allocated_blocks: int = 0
    t_first_token: Optional[int] = None
    t_done: Optional[int] = None
    alloc_stalls: int = 0

    @property
    def finished(self) -> bool:
        return self.prompt_tokens_done >= self.prompt_tokens_total and self.generated >= self.gen_tokens_target

    @property
    def tokens_materialized(self) -> int:
        return self.prompt_tokens_done + self.generated


class PagedKVAllocator:
    def __init__(self, total_blocks: int) -> None:
        self.total_blocks = total_blocks
        self.free_blocks: Set[int] = set(range(total_blocks))

    def alloc(self, n: int) -> List[int]:
        if n <= 0:
            return []
        if len(self.free_blocks) < n:
            return []
        chosen = sorted(list(self.free_blocks))[:n]
        for b in chosen:
            self.free_blocks.remove(b)
        return chosen

    def free(self, blocks: List[int]) -> None:
        for b in blocks:
            self.free_blocks.add(b)

    def free_count(self) -> int:
        return len(self.free_blocks)

    def largest_free_run(self) -> int:
        if not self.free_blocks:
            return 0
        s = sorted(self.free_blocks)
        best = 1
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    def fragmentation_ratio(self) -> float:
        free = self.free_count()
        if free == 0:
            return 0.0
        largest = self.largest_free_run()
        return 1.0 - (largest / free)


class PrefixTable:
    """Tracks shared prefix blocks by prefix key."""

    def __init__(self) -> None:
        self.refcount: Dict[int, int] = {}
        self.blocks: Dict[int, int] = {}

    def has(self, key: int) -> bool:
        return self.refcount.get(key, 0) > 0

    def acquire(self, key: int, blocks: int = 0) -> int:
        """Acquire prefix entry; returns blocks newly required (0 if reused)."""
        if self.has(key):
            self.refcount[key] += 1
            return 0
        self.refcount[key] = 1
        self.blocks[key] = blocks
        return blocks

    def release(self, key: int) -> int:
        """Release prefix entry; returns blocks that can be freed if count hits zero."""
        if not self.has(key):
            return 0
        self.refcount[key] -= 1
        if self.refcount[key] == 0:
            self.refcount.pop(key, None)
            return self.blocks.pop(key, 0)
        return 0


class Simulator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.alloc = PagedKVAllocator(args.total_blocks)
        self.prefix = PrefixTable()

        self.waiting: List[Request] = []
        self.active: List[Request] = []
        self.done: List[Request] = []

        self.block_table: Dict[int, List[int]] = {}
        self.prefix_block_table: Dict[int, List[int]] = {}

        self.frag_samples: List[float] = []
        self.util_samples: List[float] = []

    def blocks_for_tokens(self, tokens: int) -> int:
        return (tokens + self.args.block_size - 1) // self.args.block_size

    def _alloc_blocks_or_stall(self, r: Request, need_blocks: int) -> bool:
        if need_blocks <= 0:
            return True
        got = self.alloc.alloc(need_blocks)
        if len(got) != need_blocks:
            if got:
                self.alloc.free(got)
            r.alloc_stalls += 1
            return False
        self.block_table.setdefault(r.req_id, []).extend(got)
        r.allocated_blocks += need_blocks
        return True

    def admit(self) -> None:
        i = 0
        while i < len(self.waiting) and len(self.active) < self.args.max_active:
            r = self.waiting[i]

            # Allocate (or reuse) prefix blocks once at admission.
            prompt_blocks = self.blocks_for_tokens(r.prompt_tokens_total)
            required_prefix_blocks = 0
            if self.args.prefix_reuse:
                if not self.prefix.has(r.prefix_key):
                    required_prefix_blocks = prompt_blocks
            else:
                required_prefix_blocks = prompt_blocks

            if required_prefix_blocks > self.alloc.free_count():
                i += 1
                continue

            if required_prefix_blocks > 0:
                blocks = self.alloc.alloc(required_prefix_blocks)
                if len(blocks) != required_prefix_blocks:
                    if blocks:
                        self.alloc.free(blocks)
                    i += 1
                    continue
                if self.args.prefix_reuse:
                    self.prefix_block_table[r.prefix_key] = blocks
                    self.prefix.acquire(r.prefix_key, required_prefix_blocks)
                else:
                    self.block_table.setdefault(r.req_id, []).extend(blocks)
                    r.allocated_blocks += required_prefix_blocks
            else:
                self.prefix.acquire(r.prefix_key, 0)

            if self.args.prefix_reuse and self.prefix.has(r.prefix_key) and r.prefix_key in self.prefix_block_table:
                # If reusing prefix, request sees prompt as already materialized.
                r.prompt_tokens_done = r.prompt_tokens_total

            self.active.append(r)
            self.waiting.pop(i)

    def step(self, t: int) -> None:
        if not self.active:
            self.sample_allocator_state()
            return

        # Stage 1: prefill progress (for non-reused prefixes).
        for r in self.active:
            if r.prompt_tokens_done < r.prompt_tokens_total:
                r.prompt_tokens_done = min(r.prompt_tokens_total, r.prompt_tokens_done + self.args.prefill_tokens_per_step)

        # Stage 2: decode scheduling with global token budget.
        decode_budget = self.args.decode_tokens_per_step
        decodable = [r for r in self.active if r.prompt_tokens_done >= r.prompt_tokens_total and r.generated < r.gen_tokens_target]
        decodable.sort(key=lambda x: (x.gen_tokens_target - x.generated, x.t_arrival))

        for r in decodable:
            if decode_budget <= 0:
                break

            # Try to decode 1 token at a time for fairness.
            need_new_block = ((r.tokens_materialized + 1) % self.args.block_size) == 1
            if need_new_block:
                ok = self._alloc_blocks_or_stall(r, 1)
                if not ok:
                    continue

            r.generated += 1
            decode_budget -= 1
            if r.t_first_token is None:
                r.t_first_token = t

        # Stage 3: completion + free.
        still: List[Request] = []
        for r in self.active:
            if r.finished:
                r.t_done = t
                self.done.append(r)
                # Free request-private blocks.
                self.alloc.free(self.block_table.pop(r.req_id, []))
                # Release shared prefix blocks if last user.
                released = self.prefix.release(r.prefix_key)
                if released > 0 and r.prefix_key in self.prefix_block_table:
                    self.alloc.free(self.prefix_block_table.pop(r.prefix_key))
            else:
                still.append(r)
        self.active = still

        self.sample_allocator_state()

    def sample_allocator_state(self) -> None:
        self.frag_samples.append(self.alloc.fragmentation_ratio())
        util = 1.0 - (self.alloc.free_count() / self.args.total_blocks)
        self.util_samples.append(util)


def gen_arrivals(args: argparse.Namespace) -> List[Arrival]:
    random.seed(args.seed)
    out: List[Arrival] = []
    for i in range(args.requests):
        out.append(
            Arrival(
                t=random.randint(0, args.arrival_horizon),
                req_id=i,
                prompt_tokens=random.randint(args.prompt_min, args.prompt_max),
                gen_tokens=random.randint(args.gen_min, args.gen_max),
                prefix_key=random.randint(0, args.prefix_cardinality - 1),
            )
        )
    out.sort()
    return out


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = int((q / 100.0) * (len(s) - 1))
    return float(s[idx])


def run(args: argparse.Namespace) -> Dict[str, float]:
    sim = Simulator(args)
    arrivals = gen_arrivals(args)

    pending = arrivals[:]
    heapq.heapify(pending)

    t = 0
    while t < args.max_steps and (pending or sim.waiting or sim.active):
        while pending and pending[0].t <= t:
            a = heapq.heappop(pending)
            sim.waiting.append(
                Request(
                    req_id=a.req_id,
                    t_arrival=t,
                    prompt_tokens_total=a.prompt_tokens,
                    gen_tokens_target=a.gen_tokens,
                    prefix_key=a.prefix_key,
                )
            )

        sim.admit()
        sim.step(t)
        sim.admit()
        t += 1

    completed = sim.done
    if not completed:
        return {
            "completed": 0.0,
            "throughput": 0.0,
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "avg_ttft": 0.0,
            "p95_ttft": 0.0,
            "avg_alloc_stalls": 0.0,
            "avg_frag": statistics.mean(sim.frag_samples) if sim.frag_samples else 0.0,
            "p95_frag": percentile(sim.frag_samples, 95),
            "avg_util": statistics.mean(sim.util_samples) if sim.util_samples else 0.0,
        }

    latencies = [float(r.t_done - r.t_arrival) for r in completed if r.t_done is not None]
    ttft = [float(r.t_first_token - r.t_arrival) for r in completed if r.t_first_token is not None]
    stalls = [float(r.alloc_stalls) for r in completed]

    return {
        "completed": float(len(completed)),
        "throughput": len(completed) / max(1, t),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency": percentile(latencies, 95),
        "avg_ttft": statistics.mean(ttft) if ttft else 0.0,
        "p95_ttft": percentile(ttft, 95),
        "avg_alloc_stalls": statistics.mean(stalls) if stalls else 0.0,
        "avg_frag": statistics.mean(sim.frag_samples) if sim.frag_samples else 0.0,
        "p95_frag": percentile(sim.frag_samples, 95),
        "avg_util": statistics.mean(sim.util_samples) if sim.util_samples else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Lesson 19 paged KV allocator + fragmentation simulator")
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--arrival-horizon", type=int, default=240)
    p.add_argument("--prompt-min", type=int, default=128)
    p.add_argument("--prompt-max", type=int, default=2048)
    p.add_argument("--gen-min", type=int, default=64)
    p.add_argument("--gen-max", type=int, default=384)
    p.add_argument("--prefix-cardinality", type=int, default=48)

    p.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    p.add_argument("--total-blocks", type=int, default=20000)
    p.add_argument("--max-active", type=int, default=96)
    p.add_argument("--prefill-tokens-per-step", type=int, default=192)
    p.add_argument("--decode-tokens-per-step", type=int, default=128)

    p.add_argument("--prefix-reuse", action="store_true")
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--max-steps", type=int, default=30000)
    args = p.parse_args()

    s = run(args)
    print("=== Lesson 19: Paged KV Allocator Fragmentation ===")
    print(f"completed requests            : {int(s['completed'])}")
    print(f"throughput (req/step)         : {s['throughput']:.4f}")
    print(f"avg latency (steps)           : {s['avg_latency']:.2f}")
    print(f"p95 latency (steps)           : {s['p95_latency']:.2f}")
    print(f"avg TTFT (steps)              : {s['avg_ttft']:.2f}")
    print(f"p95 TTFT (steps)              : {s['p95_ttft']:.2f}")
    print(f"avg alloc stalls / request    : {s['avg_alloc_stalls']:.2f}")
    print(f"avg allocator fragmentation   : {s['avg_frag']:.4f}")
    print(f"p95 allocator fragmentation   : {s['p95_frag']:.4f}")
    print(f"avg allocator utilization     : {s['avg_util']:.4f}")


if __name__ == "__main__":
    main()
