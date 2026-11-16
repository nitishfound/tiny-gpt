"""Train a tiny GPT on a text file.

    python -m tinygpt.train --data data/shakespeare.txt --steps 3000
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .data import CharTokenizer, get_batch, train_val_split
from .model import GPT, GPTConfig


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay from max_lr down to min_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, lr: float, weight_decay: float) -> torch.optim.AdamW:
    # Weight decay on matrices (weights, embeddings) only, not on biases and LayerNorm scales.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, batch_size, iters, generator, device) -> dict[str, float]:
    model.eval()  # turns dropout off
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = [model(*get_batch(data, cfg.block_size, batch_size, generator, device))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    text = Path(args.data).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    train_data, val_data = train_val_split(tok.encode(text), args.val_fraction)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    opt = make_optimizer(model, args.lr, args.weight_decay)
    print(f"device={device} params={model.num_params():,} vocab={tok.vocab_size} train_tokens={len(train_data):,}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_val = float("inf")
    start = time.perf_counter()

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, cfg, args.batch_size, args.eval_iters, gen, device)
            elapsed = time.perf_counter() - start
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:6.1f}s")
            log_rows.append({"step": step, "train_loss": round(losses["train"], 4), "val_loss": round(losses["val"], 4)})
            if losses["val"] < best_val:  # keep only the best model (early-stopping style)
                best_val = losses["val"]
                save_checkpoint(out_dir / "ckpt.pt", model, tok, step, best_val)
        if step == args.steps:
            break

        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.lr, args.min_lr, args.warmup, args.steps)
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding gradients
        opt.step()

    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    model.eval()
    prompt = torch.tensor([tok.encode("\n")], device=device)
    sample = model.generate(prompt, 300, temperature=0.8, top_k=40, generator=gen)
    print("\n--- sample ---\n" + tok.decode(sample[0].tolist()))
    return {"best_val": best_val, "log": log_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny GPT")
    p.add_argument("--data", default="data/shakespeare.txt")
    p.add_argument("--out", default="out")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
