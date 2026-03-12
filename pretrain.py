from __future__ import annotations

from pathlib import Path

import torch

from checkpoint import save_checkpoint
from config import ModelConfig, TrainingConfig
from data import CharTokenizer, NextTokenDataset, load_text, sample_batch
from model import GPTModel
from parallelism import mixed_precision_dtype, resolve_device, set_seed


def run_pretraining(cfg: TrainingConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device()
    dtype = mixed_precision_dtype(device)

    text = load_text(cfg.train_path)
    tokenizer = CharTokenizer.train_from_text(text)
    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(token_ids, block_size=cfg.block_size)

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
    )
    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"
    checkpoint_path = out_dir / "model.pt"
    tokenizer.save(tokenizer_path)

    print(f"Training on device={device} | vocab_size={tokenizer.vocab_size} | dataset_len={len(dataset)}")

    for step in range(1, cfg.max_steps + 1):
        batch = sample_batch(dataset, batch_size=cfg.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(batch.x, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(batch.x, batch.y)
            loss.backward()
            optimizer.step()

        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.max_steps:
            print(f"step={step:04d} loss={loss.item():.4f}")

        if step % cfg.save_every == 0 or step == cfg.max_steps:
            save_checkpoint(str(checkpoint_path), model, model_cfg, step)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Pre-training complete.")
    print(f"Tokenizer:  {tokenizer_path}")
    print(f"Checkpoint: {checkpoint_path}")
