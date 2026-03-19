"""
Lesson 10: KV-cache systems engineering for long-context inference.

What this script demonstrates:
1) Paged KV-cache allocator with fixed-size blocks.
2) Continuous batching scheduler with mixed prefill/decode work.
3) Prefill/decode disaggregation mental model via two-stage token accounting.
4) Metrics: TTFT, decode throughput, and cache fragmentation.

Run:
    python3 GPU_GPT/lessons/lesson_10_kv_cache_paging_continuous_batching.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
import heapq
import math
import random
from typing import Dict, List, Tuple


@dataclass
class Request:
    req_id: int
    prompt_tokens: int
    gen_tokens: int
    arrival_step: int
    remaining_prefill: int = field(init=False)
    remaining_decode: int = field(init=False)
    first_token_step: int | None = None
    finish_step: int | None = None
    allocated_blocks: int = 0

    def __post_init__(self) -> None:
        self.remaining_prefill = self.prompt_tokens
        self.remaining_decode = self.gen_tokens

    @property
    def done(self) -> bool:
        return self.remaining_prefill == 0 and self.remaining_decode == 0


class PagedKVCache:
    def __init__(self, total_blocks: int, tokens_per_block: int) -> None:
        self.total_blocks = total_blocks
        self.tokens_per_block = tokens_per_block
        self.free_blocks = total_blocks
        self.allocations: Dict[int, int] = {}

    def blocks_needed(self, total_tokens: int) -> int:
        return math.ceil(total_tokens / self.tokens_per_block)

    def try_alloc(self, req_id: int, blocks: int) -> bool:
        if self.free_blocks < blocks:
            return False
        self.free_blocks -= blocks
        self.allocations[req_id] = self.allocations.get(req_id, 0) + blocks
        return True

    def free(self, req_id: int) -> None:
        blocks = self.allocations.pop(req_id, 0)
        self.free_blocks += blocks

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks


class ContinuousBatchScheduler:
    def __init__(
        self,
        cache: PagedKVCache,
        prefill_tokens_per_step: int,
        decode_tokens_per_step: int,
    ) -> None:
        self.cache = cache
        self.prefill_tokens_per_step = prefill_tokens_per_step
        self.decode_tokens_per_step = decode_tokens_per_step
        self.time = 0
        self.waiting: List[Tuple[int, int, Request]] = []  # (arrival, req_id, req)
        self.active_prefill: List[Request] = []
        self.active_decode: List[Request] = []
        self.completed: List[Request] = []

    def enqueue(self, req: Request) -> None:
        heapq.heappush(self.waiting, (req.arrival_step, req.req_id, req))

    def _admit_waiting(self) -> None:
        while self.waiting and self.waiting[0][0] <= self.time:
            _, _, req = heapq.heappop(self.waiting)
            total_tokens = req.prompt_tokens + req.gen_tokens
            blocks = self.cache.blocks_needed(total_tokens)
            if self.cache.try_alloc(req.req_id, blocks):
                req.allocated_blocks = blocks
                self.active_prefill.append(req)
            else:
                heapq.heappush(self.waiting, (self.time + 1, req.req_id, req))
                break

    def _run_prefill_stage(self) -> None:
        budget = self.prefill_tokens_per_step
        self.active_prefill.sort(key=lambda r: r.remaining_prefill, reverse=True)

        next_prefill: List[Request] = []
        for req in self.active_prefill:
            if budget <= 0:
                next_prefill.append(req)
                continue
            take = min(req.remaining_prefill, budget)
            req.remaining_prefill -= take
            budget -= take

            if req.remaining_prefill == 0:
                self.active_decode.append(req)
            else:
                next_prefill.append(req)

        self.active_prefill = next_prefill

    def _run_decode_stage(self) -> None:
        budget = self.decode_tokens_per_step
        self.active_decode.sort(key=lambda r: r.req_id)

        i = 0
        while budget > 0 and self.active_decode:
            req = self.active_decode[i % len(self.active_decode)]
            if req.remaining_decode > 0:
                req.remaining_decode -= 1
                budget -= 1
                if req.first_token_step is None:
                    req.first_token_step = self.time

            if req.remaining_decode == 0 and req.remaining_prefill == 0:
                req.finish_step = self.time
                self.cache.free(req.req_id)
                self.completed.append(req)
                self.active_decode = [r for r in self.active_decode if r.req_id != req.req_id]
                if not self.active_decode:
                    break
                i = i % len(self.active_decode)
            else:
                i += 1

    def step(self) -> None:
        self._admit_waiting()
        self._run_prefill_stage()
        self._run_decode_stage()
        self.time += 1

    def is_finished(self) -> bool:
        return not self.waiting and not self.active_prefill and not self.active_decode


def synthetic_workload(n: int, seed: int = 7) -> List[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        prompt = rng.randint(512, 8192)
        gen = rng.randint(128, 1024)
        arrival = rng.randint(0, 40)
        reqs.append(Request(i, prompt, gen, arrival))
    return reqs


def fragmentation_ratio(reqs: List[Request], tokens_per_block: int) -> float:
    used = 0
    reserved = 0
    for r in reqs:
        total = r.prompt_tokens + r.gen_tokens
        used += total
        reserved += math.ceil(total / tokens_per_block) * tokens_per_block
    return 1.0 - (used / reserved)


def summarize(reqs: List[Request], scheduler: ContinuousBatchScheduler) -> None:
    ttft = [r.first_token_step - r.arrival_step for r in reqs if r.first_token_step is not None]
    e2e = [r.finish_step - r.arrival_step for r in reqs if r.finish_step is not None]
    total_decoded = sum(r.gen_tokens for r in reqs)
    makespan = max((r.finish_step or 0) for r in reqs) + 1

    print("=== Lesson 10: KV-cache paging + continuous batching simulation ===")
    print(f"requests: {len(reqs)}")
    print(f"cache blocks: {scheduler.cache.total_blocks} (tokens/block={scheduler.cache.tokens_per_block})")
    print(f"prefill budget/step: {scheduler.prefill_tokens_per_step}")
    print(f"decode budget/step:  {scheduler.decode_tokens_per_step}")
    print()
    print("--- Metrics ---")
    print(f"mean TTFT (steps):      {mean(ttft):.2f}")
    print(f"p95 TTFT (steps):       {sorted(ttft)[int(0.95 * (len(ttft)-1))]:.0f}")
    print(f"mean end-to-end latency:{mean(e2e):.2f}")
    print(f"decode throughput:      {total_decoded / makespan:.2f} tokens/step")
    print(f"cache fragmentation:    {fragmentation_ratio(reqs, scheduler.cache.tokens_per_block)*100:.2f}%")
    print(f"final cache utilization:{scheduler.cache.utilization*100:.2f}%")


def main() -> None:
    reqs = synthetic_workload(n=60, seed=42)
    cache = PagedKVCache(total_blocks=2200, tokens_per_block=16)
    sched = ContinuousBatchScheduler(
        cache=cache,
        prefill_tokens_per_step=32_000,
        decode_tokens_per_step=900,
    )

    for r in reqs:
        sched.enqueue(r)

    max_steps = 50_000
    for _ in range(max_steps):
        sched.step()
        if sched.is_finished():
            break
    else:
        raise RuntimeError("Simulation did not finish. Increase max_steps or budgets.")

    summarize(reqs, sched)


if __name__ == "__main__":
    main()
