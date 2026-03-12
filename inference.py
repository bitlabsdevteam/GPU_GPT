from __future__ import annotations

import torch

from checkpoint import load_checkpoint
from config import InferenceConfig
from data import CharTokenizer
from parallelism import resolve_device


@torch.no_grad()
def run_inference(cfg: InferenceConfig) -> None:
    device = resolve_device(cfg.device)
    tokenizer = CharTokenizer.load(cfg.tokenizer_path)
    model, _, step = load_checkpoint(cfg.checkpoint_path, device=device)
    model.eval()

    prompt_ids = tokenizer.encode(cfg.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
    )
    text = tokenizer.decode(y[0].tolist())

    print(f"Loaded checkpoint from step={step}")
    print("=== PROMPT ===")
    print(cfg.prompt)
    print("=== GENERATED ===")
    print(text)
