"""Character-level tokenizer and random batch sampling."""

from __future__ import annotations

import torch


class CharTokenizer:
    """Every distinct character is one token. Tiny vocab (~65 for Shakespeare), simple, lossless."""

    def __init__(self, chars: list[str]):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def train_val_split(ids: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of text. y is x shifted left by one: the "next token" at every position.

    x = "To be, or"   ->   y = "o be, or "
    """
    if len(data) <= block_size:
        raise ValueError("dataset is smaller than block_size")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts.tolist()])
    return x.to(device), y.to(device)
