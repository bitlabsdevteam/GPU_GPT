"""
Lesson 11: Speculative decoding with draft-target verification.

This lesson demonstrates a production-relevant inference optimization:
- A small draft model proposes K tokens in one shot.
- A larger target model verifies those proposals in parallel.
- Accepted prefix tokens are committed; on first rejection, we resample from the
  target at that position to preserve exact target distribution.

What to look at:
1) Acceptance rate vs speedup tradeoff.
2) Impact of draft quality (temperature/noise) on throughput.
3) Adaptive K policy driven by rolling acceptance.

Run:
    python3 GPU_GPT/lessons/lesson_11_speculative_decoding.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 512
    d_model: int = 128
    hidden: int = 256
    context: int = 64
    prompt_len: int = 16
    max_new_tokens: int = 96

    # speculative decode controls
    init_k: int = 6
    min_k: int = 2
    max_k: int = 12
    target_accept_hi: float = 0.80
    target_accept_lo: float = 0.50

    seed: int = 42
    device: str = "cpu"


class TinyLM(nn.Module):
    """Small causal LM: token embedding + MLP + LM head.

    This is intentionally tiny and untrained; we synthesize a "target" and create a
    noisy "draft" copy to emulate quality differences.
    """

    def __init__(self, vocab_size: int, d_model: int, hidden: int, context: int):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        b, t = x.shape
        pos_ids = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok(x) + self.pos(pos_ids)
        h = self.ff(h)
        return self.lm_head(h)  # [B, T, V]

    @torch.no_grad()
    def next_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, -1, :]


@torch.no_grad()
def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def draft_propose(
    draft: TinyLM,
    prefix: torch.Tensor,
    k: int,
    temperature: float,
) -> Tuple[List[int], List[torch.Tensor]]:
    """Autoregressively propose K tokens with the draft model.

    Returns:
      tokens: list[int] length K
      draft_probs: probability vectors q_i for each proposed token position i
    """
    toks: List[int] = []
    q_probs: List[torch.Tensor] = []
    cur = prefix.clone()

    for _ in range(k):
        logits = draft.next_logits(cur.unsqueeze(0)).squeeze(0)  # [V]
        q = F.softmax(logits / max(temperature, 1e-5), dim=-1)
        y = torch.multinomial(q, 1).item()
        toks.append(y)
        q_probs.append(q)
        cur = torch.cat([cur, torch.tensor([y], device=cur.device, dtype=cur.dtype)])

    return toks, q_probs


@torch.no_grad()
def target_verify_prefix(
    target: TinyLM,
    prefix: torch.Tensor,
    draft_tokens: List[int],
    draft_probs: List[torch.Tensor],
    temperature: float,
) -> Tuple[List[int], bool, int]:
    """Verify draft tokens against target using acceptance-rejection.

    Accept token y_i with probability min(1, p_i(y_i)/q_i(y_i)).
    On first rejection at position i:
      sample from corrected distribution proportional to [p_i - q_i]_+.

    Returns:
      committed_tokens: accepted tokens + one corrective sample if rejection occurred
      all_accepted: whether full K proposal accepted
      accepted_count: number of accepted draft tokens before rejection
    """
    if len(draft_tokens) == 0:
        return [], True, 0

    k = len(draft_tokens)
    y = torch.tensor(draft_tokens, device=prefix.device, dtype=prefix.dtype)

    # Build one forward pass on prefix + full proposal for parallel target conditionals.
    full = torch.cat([prefix, y])
    logits = target(full.unsqueeze(0)).squeeze(0)  # [T_full, V]

    committed: List[int] = []

    for i in range(k):
        pos = prefix.numel() - 1 + i
        p = F.softmax(logits[pos] / max(temperature, 1e-5), dim=-1)  # target next-token dist
        q = draft_probs[i]
        token = draft_tokens[i]

        accept_prob = torch.minimum(torch.tensor(1.0, device=p.device), p[token] / q[token].clamp_min(1e-9))
        u = torch.rand((), device=p.device)

        if u <= accept_prob:
            committed.append(token)
            continue

        # First rejection => corrective sample from normalized (p - q)_+
        corr = torch.clamp(p - q, min=0.0)
        z = corr.sum()
        if z <= 1e-12:
            # Fallback if numerically degenerate.
            replacement = torch.multinomial(p, 1).item()
        else:
            replacement = torch.multinomial(corr / z, 1).item()

        committed.append(replacement)
        return committed, False, i

    # If all accepted, append one extra token sampled from target at the "+1" position.
    final_prefix = torch.cat([prefix, torch.tensor(committed, device=prefix.device, dtype=prefix.dtype)])
    next_p = F.softmax(target.next_logits(final_prefix.unsqueeze(0)).squeeze(0) / max(temperature, 1e-5), dim=-1)
    tail = torch.multinomial(next_p, 1).item()
    committed.append(tail)
    return committed, True, k


@torch.no_grad()
def greedy_decode(model: TinyLM, prefix: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    out = prefix.clone()
    for _ in range(max_new_tokens):
        logits = model.next_logits(out.unsqueeze(0)).squeeze(0)
        nxt = sample_from_logits(logits, temperature=temperature)
        out = torch.cat([out, nxt.view(1)])
    return out


@torch.no_grad()
def speculative_decode(
    draft: TinyLM,
    target: TinyLM,
    prefix: torch.Tensor,
    max_new_tokens: int,
    cfg: Config,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    out = prefix.clone()
    proposed = 0
    accepted = 0
    target_steps = 0
    k = cfg.init_k

    while out.numel() < prefix.numel() + max_new_tokens:
        remain = prefix.numel() + max_new_tokens - out.numel()
        cur_k = min(k, remain)

        # draft autoregressive proposals
        toks, q_probs = draft_propose(draft, out, cur_k, temperature=temperature)
        proposed += len(toks)

        committed, all_acc, accepted_count = target_verify_prefix(
            target, out, toks, q_probs, temperature=temperature
        )
        accepted += accepted_count
        target_steps += 1

        out = torch.cat([out, torch.tensor(committed, device=out.device, dtype=out.dtype)])
        out = out[: prefix.numel() + max_new_tokens]

        # Simple adaptive-K controller.
        rolling_acc = accepted / max(proposed, 1)
        if rolling_acc >= cfg.target_accept_hi:
            k = min(cfg.max_k, k + 1)
        elif rolling_acc <= cfg.target_accept_lo:
            k = max(cfg.min_k, k - 1)

    stats = {
        "proposed": proposed,
        "accepted": accepted,
        "accept_rate": accepted / max(proposed, 1),
        "target_verify_steps": target_steps,
        "avg_committed_per_verify": max_new_tokens / max(target_steps, 1),
    }
    return out, stats


def clone_as_noisy_draft(target: TinyLM, noise_scale: float = 0.08) -> TinyLM:
    draft = TinyLM(
        vocab_size=target.lm_head.out_features,
        d_model=target.tok.embedding_dim,
        hidden=target.ff[0].out_features,
        context=target.context,
    )
    draft.load_state_dict(target.state_dict())
    for p in draft.parameters():
        p.add_(noise_scale * torch.randn_like(p))
    return draft


def main() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)

    target = TinyLM(cfg.vocab_size, cfg.d_model, cfg.hidden, cfg.context).to(cfg.device)
    draft = clone_as_noisy_draft(target, noise_scale=0.10).to(cfg.device)

    prefix = torch.randint(0, cfg.vocab_size, (cfg.prompt_len,), device=cfg.device)

    t0 = perf_counter()
    baseline = greedy_decode(target, prefix, cfg.max_new_tokens, temperature=1.0)
    t1 = perf_counter()

    t2 = perf_counter()
    spec_out, stats = speculative_decode(
        draft=draft,
        target=target,
        prefix=prefix,
        max_new_tokens=cfg.max_new_tokens,
        cfg=cfg,
        temperature=1.0,
    )
    t3 = perf_counter()

    base_time = t1 - t0
    spec_time = t3 - t2

    print("=== Lesson 11: Speculative Decoding ===")
    print(f"Prompt length: {cfg.prompt_len} | New tokens: {cfg.max_new_tokens}")
    print(f"Baseline (target-only) wall time: {base_time:.4f}s")
    print(f"Speculative wall time:           {spec_time:.4f}s")
    if spec_time > 0:
        print(f"Measured speedup:                {base_time / spec_time:.2f}x")

    print("\n--- Speculative stats ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nSanity:")
    print(f"Baseline output tokens:    {baseline.numel()}")
    print(f"Speculative output tokens: {spec_out.numel()}")
    print(
        "Note: token sequences differ because both paths sample stochastically, "
        "but speculative decode preserves target distribution by construction."
    )


if __name__ == "__main__":
    main()
