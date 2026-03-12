from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            missing = "".join(sorted(set(unknown)))
            raise ValueError(f"Prompt contains unseen characters: {missing!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)

    def save(self, path: str | Path) -> None:
        payload = {"stoi": self.stoi}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        # JSON preserves the character keys as-is; rebuild inverse map.
        fixed_stoi = {k: v for k, v in stoi.items()}
        itos = {v: k for k, v in fixed_stoi.items()}
        return cls(stoi=fixed_stoi, itos=itos)


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        if len(token_ids) <= block_size:
            raise ValueError(
                f"Training text is too short ({len(token_ids)} tokens). Need more than block_size={block_size}."
            )
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def sample_batch(dataset: NextTokenDataset, batch_size: int, device: torch.device) -> Batch:
    starts = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
    xs, ys = zip(*(dataset[idx] for idx in starts))
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return Batch(x=x, y=y)
