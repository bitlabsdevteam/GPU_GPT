"""
Lesson 08: Expert Parallelism (MoE) with Top-2 routing and capacity control.

This script is intentionally compact but practically relevant:
- Token-wise Top-2 routing over experts
- Capacity factor + token dropping accounting
- Auxiliary load-balancing loss (Switch-style)
- Per-step expert utilization diagnostics

Run:
  python GPU_GPT/lessons/lesson_08_moe_expert_parallelism_top2.py
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 256
    d_ff: int = 1024
    seq_len: int = 128
    batch_size: int = 16
    n_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_coef: float = 1e-2
    lr: float = 3e-4
    steps: int = 40
    vocab_size: int = 8192


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Top2MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, capacity_factor: float):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # x: [T, D]
        T, D = x.shape
        logits = self.router(x)                                 # [T, E]
        probs = F.softmax(logits, dim=-1)                       # [T, E]

        top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)    # [T,2], [T,2]
        # Renormalize on selected experts.
        gates = top2_vals / top2_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Capacity per expert per forward pass.
        capacity = int(self.capacity_factor * (2 * T) / self.n_experts)
        capacity = max(capacity, 1)

        out = torch.zeros_like(x)
        accepted = torch.zeros(T, 2, dtype=torch.bool, device=x.device)
        per_expert_assigned = torch.zeros(self.n_experts, device=x.device)
        per_expert_used = torch.zeros(self.n_experts, device=x.device)

        # Greedy expert queues to emulate dispatch buffers.
        for e in range(self.n_experts):
            pick_mask = top2_idx.eq(e)                          # [T,2]
            token_pos, route_pos = torch.where(pick_mask)
            per_expert_assigned[e] = token_pos.numel()

            if token_pos.numel() == 0:
                continue

            keep = min(token_pos.numel(), capacity)
            token_pos = token_pos[:keep]
            route_pos = route_pos[:keep]
            per_expert_used[e] = keep
            accepted[token_pos, route_pos] = True

            x_e = x[token_pos]                                  # [keep, D]
            y_e = self.experts[e](x_e)                          # [keep, D]
            w_e = gates[token_pos, route_pos].unsqueeze(-1)     # [keep,1]
            out[token_pos] += w_e * y_e

        # Tokens with zero accepted routes are dropped for this layer.
        accepted_count = accepted.sum(dim=-1)                   # [T]
        dropped_mask = accepted_count.eq(0)
        out[dropped_mask] = x[dropped_mask]                     # residual fallback

        # Load balancing auxiliary loss (Switch-style variant).
        # importance: average routing probability per expert
        # load: fraction of selected routes per expert (before capacity truncation)
        importance = probs.mean(dim=0)                          # [E]
        load = top2_idx.reshape(-1).bincount(minlength=self.n_experts).float()
        load = load / load.sum().clamp_min(1e-9)
        aux_loss = self.n_experts * torch.sum(importance * load)

        stats = {
            "capacity": capacity,
            "drop_rate": dropped_mask.float().mean().item(),
            "assigned_per_expert": per_expert_assigned.detach().cpu(),
            "used_per_expert": per_expert_used.detach().cpu(),
            "importance": importance.detach().cpu(),
        }
        return out, aux_loss, stats


class TinyMoEBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.moe = Top2MoE(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.capacity_factor)

    def forward(self, x: torch.Tensor):
        y, aux_loss, stats = self.moe(self.ln(x))
        return x + y, aux_loss, stats


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.block = TinyMoEBlock(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor):
        h = self.embed(ids)                       # [B,S,D]
        h2d = h.reshape(-1, h.size(-1))          # [T,D]
        h2d, aux_loss, stats = self.block(h2d)
        h = h2d.reshape_as(h)
        logits = self.head(h)
        return logits, aux_loss, stats


def main():
    torch.manual_seed(42)
    cfg = Config()
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    tokens_per_step = cfg.batch_size * cfg.seq_len
    print("=== Lesson 08: MoE Expert Parallelism (Top-2) ===")
    print(
        f"experts={cfg.n_experts} top_k={cfg.top_k} capacity_factor={cfg.capacity_factor} "
        f"tokens/step={tokens_per_step}"
    )

    for step in range(1, cfg.steps + 1):
        x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

        opt.zero_grad(set_to_none=True)
        logits, aux_loss, stats = model(x)
        ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss = ce + cfg.aux_loss_coef * aux_loss
        loss.backward()
        opt.step()

        if step % 5 == 0 or step == 1:
            used = stats["used_per_expert"].tolist()
            print(
                f"step={step:03d} loss={loss.item():.4f} ce={ce.item():.4f} "
                f"aux={aux_loss.item():.4f} cap={stats['capacity']} drop={stats['drop_rate']*100:.2f}%"
            )
            print(f"  used_per_expert={used}")


if __name__ == "__main__":
    main()
