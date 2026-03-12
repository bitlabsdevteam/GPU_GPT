from __future__ import annotations

from pathlib import Path

import torch

from config import ModelConfig
from model import GPTModel


def save_checkpoint(path: str, model: GPTModel, model_config: ModelConfig, step: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[GPTModel, ModelConfig, int]:
    payload = torch.load(path, map_location=device)
    model_config = ModelConfig(**payload["model_config"])
    model = GPTModel(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    step = int(payload.get("step", 0))
    return model, model_config, step
