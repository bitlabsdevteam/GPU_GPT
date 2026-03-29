"""
Lesson 20: Prefix KV Cache with Radix Tree + Page-Aware Refcounting.

Why this matters:
- After paged KV allocation, the next serving win is avoiding duplicate prefill work.
- Prefix caching reuses KV for shared prompt prefixes across requests.
- Real systems need a safe eviction policy that respects active decodes.

This simulation demonstrates:
1) A radix-tree-like prefix index over token sequences.
2) Paged KV accounting (logical pages, capacity budget).
3) Refcounted pin/unpin for active requests.
4) Greedy eviction of cold, unpinned prefixes.

Run:
  python3 lessons/lesson_20_prefix_cache_radix_tree.py \
      --page-size 16 --max-pages 96 --requests 2000 --seed 7
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


TokenSeq = Tuple[int, ...]


@dataclass
class PrefixEntry:
    prefix: TokenSeq
    pages: int
    bytes_used: int
    last_touch: int = 0
    hits: int = 0
    pins: int = 0

    @property
    def evictable(self) -> bool:
        return self.pins == 0


class PrefixCache:
    def __init__(self, page_size_tokens: int, max_pages: int, bytes_per_token_kv: int = 2048):
        self.page_size_tokens = page_size_tokens
        self.max_pages = max_pages
        self.bytes_per_token_kv = bytes_per_token_kv

        self.used_pages = 0
        self.clock = 0

        # Simplified radix index: store all prefixes directly for clarity.
        self.entries: Dict[TokenSeq, PrefixEntry] = {}

        self.metrics = {
            "lookups": 0,
            "hit_tokens": 0,
            "saved_prefill_tokens": 0,
            "misses": 0,
            "evictions": 0,
            "evicted_pages": 0,
            "admissions": 0,
            "rejections_no_space": 0,
        }

    def _pages_for_tokens(self, n_tokens: int) -> int:
        return (n_tokens + self.page_size_tokens - 1) // self.page_size_tokens

    def _bytes_for_tokens(self, n_tokens: int) -> int:
        return n_tokens * self.bytes_per_token_kv

    def _touch(self, e: PrefixEntry) -> None:
        self.clock += 1
        e.last_touch = self.clock

    def longest_prefix_hit(self, seq: TokenSeq) -> Optional[PrefixEntry]:
        self.metrics["lookups"] += 1
        # Walk from longest -> shortest for best reuse.
        for i in range(len(seq), 0, -1):
            pref = seq[:i]
            e = self.entries.get(pref)
            if e is not None:
                e.hits += 1
                self._touch(e)
                self.metrics["hit_tokens"] += len(pref)
                self.metrics["saved_prefill_tokens"] += len(pref)
                return e
        self.metrics["misses"] += 1
        return None

    def pin(self, prefix: TokenSeq) -> bool:
        e = self.entries.get(prefix)
        if e is None:
            return False
        e.pins += 1
        self._touch(e)
        return True

    def unpin(self, prefix: TokenSeq) -> bool:
        e = self.entries.get(prefix)
        if e is None or e.pins == 0:
            return False
        e.pins -= 1
        self._touch(e)
        return True

    def admit(self, prefix: TokenSeq) -> bool:
        if not prefix:
            return False
        if prefix in self.entries:
            self._touch(self.entries[prefix])
            return True

        need_pages = self._pages_for_tokens(len(prefix))
        if need_pages > self.max_pages:
            self.metrics["rejections_no_space"] += 1
            return False

        self._ensure_space(need_pages)
        if self.used_pages + need_pages > self.max_pages:
            self.metrics["rejections_no_space"] += 1
            return False

        e = PrefixEntry(
            prefix=prefix,
            pages=need_pages,
            bytes_used=self._bytes_for_tokens(len(prefix)),
        )
        self._touch(e)
        self.entries[prefix] = e
        self.used_pages += need_pages
        self.metrics["admissions"] += 1
        return True

    def _ensure_space(self, need_pages: int) -> None:
        if self.used_pages + need_pages <= self.max_pages:
            return

        # LRU over unpinned entries only.
        victims = sorted(
            (e for e in self.entries.values() if e.evictable),
            key=lambda x: x.last_touch,
        )
        for e in victims:
            if self.used_pages + need_pages <= self.max_pages:
                break
            self._evict(e.prefix)

    def _evict(self, prefix: TokenSeq) -> None:
        e = self.entries.pop(prefix, None)
        if e is None:
            return
        self.used_pages -= e.pages
        self.metrics["evictions"] += 1
        self.metrics["evicted_pages"] += e.pages

    def stats(self) -> Dict[str, float]:
        total_entries = len(self.entries)
        total_tokens = sum(len(p) for p in self.entries)
        pinned = sum(1 for e in self.entries.values() if e.pins > 0)

        hit_rate = 1.0 - (self.metrics["misses"] / max(1, self.metrics["lookups"]))
        avg_hit = self.metrics["hit_tokens"] / max(1, (self.metrics["lookups"] - self.metrics["misses"]))
        util = self.used_pages / max(1, self.max_pages)

        return {
            "entries": float(total_entries),
            "resident_tokens": float(total_tokens),
            "pinned_entries": float(pinned),
            "used_pages": float(self.used_pages),
            "max_pages": float(self.max_pages),
            "page_utilization": util,
            "lookups": float(self.metrics["lookups"]),
            "hit_rate": hit_rate,
            "avg_hit_tokens": avg_hit,
            "saved_prefill_tokens": float(self.metrics["saved_prefill_tokens"]),
            "admissions": float(self.metrics["admissions"]),
            "evictions": float(self.metrics["evictions"]),
            "evicted_pages": float(self.metrics["evicted_pages"]),
            "reject_no_space": float(self.metrics["rejections_no_space"]),
        }


def build_prompt_family(rng: random.Random, families: int, family_prefix: int, tail_min: int, tail_max: int) -> List[TokenSeq]:
    prompts: List[TokenSeq] = []
    for fam in range(families):
        root = tuple([10_000 + fam] + [rng.randint(100, 9999) for _ in range(family_prefix - 1)])
        # Each family gets many variants sharing the root.
        for _ in range(32):
            tail = tuple(rng.randint(100, 9999) for _ in range(rng.randint(tail_min, tail_max)))
            prompts.append(root + tail)
    return prompts


def simulate(args: argparse.Namespace) -> Dict[str, float]:
    rng = random.Random(args.seed)
    cache = PrefixCache(page_size_tokens=args.page_size, max_pages=args.max_pages)

    prompts = build_prompt_family(
        rng=rng,
        families=args.families,
        family_prefix=args.family_prefix,
        tail_min=args.tail_min,
        tail_max=args.tail_max,
    )

    # Keep some active requests pinned for realism.
    active: List[TokenSeq] = []

    for step in range(args.requests):
        req = rng.choice(prompts)

        hit = cache.longest_prefix_hit(req)
        if hit is not None:
            cache.pin(hit.prefix)
            active.append(hit.prefix)

        # Admit a moderately long prefix for future reuse.
        # In production this is often chosen at chunk boundaries.
        cutoff = min(len(req), rng.randint(args.admit_min, args.admit_max))
        cache.admit(req[:cutoff])

        # Randomly finish some active decodes (unpin).
        for _ in range(rng.randint(0, args.max_completions_per_step)):
            if not active:
                break
            i = rng.randrange(len(active))
            p = active.pop(i)
            cache.unpin(p)

        # Periodically re-touch hot prefixes to mimic temporal locality.
        if step % 20 == 0:
            for _ in range(3):
                hot = rng.choice(prompts)
                cache.longest_prefix_hit(hot)

    # Drain leftovers.
    for p in active:
        cache.unpin(p)

    return cache.stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--max-pages", type=int, default=96)
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--families", type=int, default=24)
    ap.add_argument("--family-prefix", type=int, default=20)
    ap.add_argument("--tail-min", type=int, default=24)
    ap.add_argument("--tail-max", type=int, default=128)
    ap.add_argument("--admit-min", type=int, default=16)
    ap.add_argument("--admit-max", type=int, default=96)
    ap.add_argument("--max-completions-per-step", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s = simulate(args)

    print("=== Lesson 20: Prefix Cache + Radix Index (Simulation) ===")
    print(f"entries               : {int(s['entries'])}")
    print(f"resident_tokens       : {int(s['resident_tokens'])}")
    print(f"pinned_entries        : {int(s['pinned_entries'])}")
    print(f"pages                 : {int(s['used_pages'])}/{int(s['max_pages'])} ({100*s['page_utilization']:.1f}%)")
    print(f"lookups               : {int(s['lookups'])}")
    print(f"hit_rate              : {100*s['hit_rate']:.2f}%")
    print(f"avg_hit_tokens        : {s['avg_hit_tokens']:.2f}")
    print(f"saved_prefill_tokens  : {int(s['saved_prefill_tokens'])}")
    print(f"admissions            : {int(s['admissions'])}")
    print(f"evictions             : {int(s['evictions'])}")
    print(f"evicted_pages         : {int(s['evicted_pages'])}")
    print(f"reject_no_space       : {int(s['reject_no_space'])}")


if __name__ == "__main__":
    main()
