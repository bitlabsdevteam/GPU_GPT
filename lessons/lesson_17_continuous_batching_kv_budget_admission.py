#!/usr/bin/env python3
"""
Lesson 17: Continuous batching with KV-budget admission control.

This script simulates an LLM serving scheduler where requests arrive over time and
compete for a finite KV-cache budget. It demonstrates practical policies used in
production inference stacks:

1) Prefix-aware admission (reuse prompt cache when possible)
2) Token-budgeted decode scheduling (max tokens per step)
3) Preemption/eviction under memory pressure (largest-KV-first)
4) Queueing metrics: throughput, latency, TTFT, p95 tail

No GPU required: this is a systems simulator for policy tuning.
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(order=True)
class Arrival:
    t: int
    req_id: int = field(compare=False)
    prompt_tokens: int = field(compare=False)
    gen_tokens: int = field(compare=False)
    prefix_key: int = field(compare=False)


@dataclass
class RequestState:
    req_id: int
    t_arrival: int
    prompt_tokens: int
    gen_tokens_target: int
    prefix_key: int
    prefill_done: bool = False
    generated: int = 0
    t_first_token: Optional[int] = None
    t_done: Optional[int] = None
    evictions: int = 0

    @property
    def finished(self) -> bool:
        return self.prefill_done and self.generated >= self.gen_tokens_target


class PrefixCache:
    """Tracks reusable prompt prefixes already materialized in KV memory."""

    def __init__(self) -> None:
        self._prefix_refcount: Dict[int, int] = {}

    def has(self, prefix_key: int) -> bool:
        return self._prefix_refcount.get(prefix_key, 0) > 0

    def acquire(self, prefix_key: int) -> None:
        self._prefix_refcount[prefix_key] = self._prefix_refcount.get(prefix_key, 0) + 1

    def release(self, prefix_key: int) -> None:
        cur = self._prefix_refcount.get(prefix_key, 0)
        if cur <= 1:
            self._prefix_refcount.pop(prefix_key, None)
        else:
            self._prefix_refcount[prefix_key] = cur - 1


class Scheduler:
    def __init__(
        self,
        kv_capacity_tokens: int,
        max_decode_tokens_per_step: int,
        prefill_chunk_tokens: int,
        enable_prefix_reuse: bool,
        enable_eviction: bool,
    ) -> None:
        self.kv_capacity = kv_capacity_tokens
        self.max_decode_tokens_per_step = max_decode_tokens_per_step
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.enable_prefix_reuse = enable_prefix_reuse
        self.enable_eviction = enable_eviction

        self.waiting: List[RequestState] = []
        self.active: List[RequestState] = []
        self.done: List[RequestState] = []

        self.prefix_cache = PrefixCache()
        self.cur_kv_tokens = 0

    def kv_footprint(self, r: RequestState) -> int:
        # For this simulation, KV grows with prompt + generated tokens.
        # Prefix reuse can eliminate prompt KV expansion at admission.
        prompt_cost = 0 if (self.enable_prefix_reuse and self.prefix_cache.has(r.prefix_key)) else r.prompt_tokens
        decode_cost = r.generated
        return prompt_cost + decode_cost

    def full_kv_footprint_with_next_token(self, r: RequestState) -> int:
        prompt_cost = 0 if (self.enable_prefix_reuse and self.prefix_cache.has(r.prefix_key)) else r.prompt_tokens
        return prompt_cost + r.generated + 1

    def recompute_cur_kv(self) -> None:
        self.cur_kv_tokens = sum(self.kv_footprint(r) for r in self.active)

    def maybe_admit(self) -> None:
        # FIFO admission from waiting queue while capacity allows.
        i = 0
        while i < len(self.waiting):
            r = self.waiting[i]
            need = r.prompt_tokens if not (self.enable_prefix_reuse and self.prefix_cache.has(r.prefix_key)) else 0
            if self.cur_kv_tokens + need <= self.kv_capacity:
                self.active.append(r)
                self.cur_kv_tokens += need
                self.prefix_cache.acquire(r.prefix_key)
                self.waiting.pop(i)
            else:
                if self.enable_eviction and self.active:
                    self.evict_one()
                    continue
                i += 1

    def evict_one(self) -> None:
        # Evict the active request with largest KV footprint.
        victim = max(self.active, key=self.kv_footprint)
        self.active.remove(victim)
        victim.evictions += 1
        self.prefix_cache.release(victim.prefix_key)
        self.waiting.insert(0, victim)
        self.recompute_cur_kv()

    def run_step(self, t: int) -> None:
        if not self.active:
            return

        # Stage 1: prefill in chunks
        for r in self.active:
            if not r.prefill_done:
                r.prompt_tokens -= min(self.prefill_chunk_tokens, r.prompt_tokens)
                if r.prompt_tokens <= 0:
                    r.prompt_tokens = 0
                    r.prefill_done = True

        # Stage 2: decode with token budget across prefill-complete requests
        decodable = [r for r in self.active if r.prefill_done and not r.finished]
        if decodable:
            # Shortest-remaining-generation first to tighten tails.
            decodable.sort(key=lambda r: r.gen_tokens_target - r.generated)
            budget = self.max_decode_tokens_per_step
            for r in decodable:
                if budget <= 0:
                    break
                # Check if adding a token would exceed KV capacity.
                extra = self.full_kv_footprint_with_next_token(r) - self.kv_footprint(r)
                if self.cur_kv_tokens + extra <= self.kv_capacity:
                    r.generated += 1
                    self.cur_kv_tokens += extra
                    budget -= 1
                    if r.t_first_token is None:
                        r.t_first_token = t
                elif self.enable_eviction and len(self.active) > 1:
                    self.evict_one()
                    break

        # Stage 3: complete finished requests
        still_active: List[RequestState] = []
        for r in self.active:
            if r.finished:
                r.t_done = t
                self.done.append(r)
                self.prefix_cache.release(r.prefix_key)
            else:
                still_active.append(r)
        self.active = still_active
        self.recompute_cur_kv()


def generate_arrivals(
    n_requests: int,
    max_t: int,
    prompt_min: int,
    prompt_max: int,
    gen_min: int,
    gen_max: int,
    prefix_cardinality: int,
    seed: int,
) -> List[Arrival]:
    random.seed(seed)
    arrivals: List[Arrival] = []
    for req_id in range(n_requests):
        t = random.randint(0, max_t)
        arrivals.append(
            Arrival(
                t=t,
                req_id=req_id,
                prompt_tokens=random.randint(prompt_min, prompt_max),
                gen_tokens=random.randint(gen_min, gen_max),
                prefix_key=random.randint(0, prefix_cardinality - 1),
            )
        )
    arrivals.sort()
    return arrivals


def simulate(args: argparse.Namespace) -> Dict[str, float]:
    arrivals = generate_arrivals(
        n_requests=args.requests,
        max_t=args.arrival_horizon,
        prompt_min=args.prompt_min,
        prompt_max=args.prompt_max,
        gen_min=args.gen_min,
        gen_max=args.gen_max,
        prefix_cardinality=args.prefix_cardinality,
        seed=args.seed,
    )

    sched = Scheduler(
        kv_capacity_tokens=args.kv_capacity,
        max_decode_tokens_per_step=args.decode_budget,
        prefill_chunk_tokens=args.prefill_chunk,
        enable_prefix_reuse=args.prefix_reuse,
        enable_eviction=args.eviction,
    )

    pending: List[Arrival] = arrivals[:]
    heapq.heapify(pending)

    t = 0
    max_steps = args.max_steps
    while t < max_steps and (pending or sched.waiting or sched.active):
        while pending and pending[0].t <= t:
            a = heapq.heappop(pending)
            sched.waiting.append(
                RequestState(
                    req_id=a.req_id,
                    t_arrival=t,
                    prompt_tokens=a.prompt_tokens,
                    gen_tokens_target=a.gen_tokens,
                    prefix_key=a.prefix_key,
                )
            )

        sched.maybe_admit()
        sched.run_step(t)
        sched.maybe_admit()
        t += 1

    completed = sched.done
    if not completed:
        return {
            "completed": 0,
            "throughput_req_per_step": 0.0,
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "avg_ttft": 0.0,
            "p95_ttft": 0.0,
            "avg_evictions": 0.0,
        }

    latencies = [r.t_done - r.t_arrival for r in completed if r.t_done is not None]
    ttfts = [r.t_first_token - r.t_arrival for r in completed if r.t_first_token is not None]
    evictions = [r.evictions for r in completed]

    def p95(xs: List[int]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = int(0.95 * (len(s) - 1))
        return float(s[idx])

    return {
        "completed": float(len(completed)),
        "throughput_req_per_step": len(completed) / max(1, t),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency": p95(latencies),
        "avg_ttft": statistics.mean(ttfts) if ttfts else 0.0,
        "p95_ttft": p95(ttfts),
        "avg_evictions": statistics.mean(evictions) if evictions else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Lesson 17 KV-budgeted continuous batching simulator")
    p.add_argument("--requests", type=int, default=256)
    p.add_argument("--arrival-horizon", type=int, default=200)
    p.add_argument("--prompt-min", type=int, default=64)
    p.add_argument("--prompt-max", type=int, default=1024)
    p.add_argument("--gen-min", type=int, default=32)
    p.add_argument("--gen-max", type=int, default=256)
    p.add_argument("--prefix-cardinality", type=int, default=32, help="Smaller => more prefix reuse")
    p.add_argument("--kv-capacity", type=int, default=20000)
    p.add_argument("--decode-budget", type=int, default=96, help="Max decode tokens generated per scheduler step")
    p.add_argument("--prefill-chunk", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prefix-reuse", action="store_true")
    p.add_argument("--eviction", action="store_true")
    args = p.parse_args()

    stats = simulate(args)
    print("=== Lesson 17: Continuous Batching + KV Budget Admission ===")
    print(f"completed requests       : {int(stats['completed'])}")
    print(f"throughput (req/step)   : {stats['throughput_req_per_step']:.4f}")
    print(f"avg latency (steps)     : {stats['avg_latency']:.2f}")
    print(f"p95 latency (steps)     : {stats['p95_latency']:.2f}")
    print(f"avg TTFT (steps)        : {stats['avg_ttft']:.2f}")
    print(f"p95 TTFT (steps)        : {stats['p95_ttft']:.2f}")
    print(f"avg evictions/request   : {stats['avg_evictions']:.2f}")


if __name__ == "__main__":
    main()
