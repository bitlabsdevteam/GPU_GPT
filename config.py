from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    train_path: str
    out_dir: str = "artifacts"
    batch_size: int = 16
    block_size: int = 128
    max_steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_interval: int = 20
    save_every: int = 100
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    seed: int = 42


@dataclass
class InferenceConfig:
    checkpoint_path: str
    tokenizer_path: str
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 20
    device: str | None = None
