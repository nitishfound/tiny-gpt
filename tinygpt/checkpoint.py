"""Save and load a trained model together with its config and tokenizer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .data import CharTokenizer
from .model import GPT, GPTConfig


def save_checkpoint(path: str | Path, model: GPT, tokenizer: CharTokenizer, step: int, val_loss: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.cfg),
            "chars": tokenizer.chars,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, CharTokenizer, dict]:
    # weights_only=True: never unpickle arbitrary objects from a file (safe loading).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, CharTokenizer(ckpt["chars"]), {"step": ckpt["step"], "val_loss": ckpt["val_loss"]}
