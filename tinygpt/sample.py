"""Generate text from a trained checkpoint.

    python -m tinygpt.sample --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top-k 40
"""

from __future__ import annotations

import argparse
import time

import torch

from .checkpoint import load_checkpoint


def main(argv: list[str] | None = None) -> str:
    p = argparse.ArgumentParser(description="Sample from a tiny GPT")
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true", help="disable the KV cache (slower)")
    args = p.parse_args(argv)

    model, tok, meta = load_checkpoint(args.ckpt)
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.tensor([tok.encode(args.prompt)])
    start = time.perf_counter()
    out = model.generate(
        idx, args.tokens, args.temperature, args.top_k, args.top_p, use_cache=not args.no_cache, generator=gen
    )
    elapsed = time.perf_counter() - start
    text = tok.decode(out[0].tolist())
    print(text)
    print(f"\n[{args.tokens} tokens in {elapsed:.2f}s = {args.tokens / elapsed:.0f} tok/s | step {meta['step']}, val loss {meta['val_loss']:.3f}]")
    return text


if __name__ == "__main__":
    main()
