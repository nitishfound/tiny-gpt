"""Compare generation speed with and without the KV cache.

    python scripts/bench_kv_cache.py --ckpt out/ckpt.pt --tokens 500
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinygpt.checkpoint import load_checkpoint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="out/ckpt.pt")
p.add_argument("--tokens", type=int, default=500)
args = p.parse_args()

model, tok, _ = load_checkpoint(args.ckpt)
prompt = torch.tensor([tok.encode("ROMEO:\n")])
torch.set_num_threads(max(1, torch.get_num_threads()))

results = {}
for use_cache in (False, True):
    model.generate(prompt, 20, temperature=0, use_cache=use_cache)  # warm-up
    start = time.perf_counter()
    out = model.generate(prompt, args.tokens, temperature=0, use_cache=use_cache)
    elapsed = time.perf_counter() - start
    results[use_cache] = (out, args.tokens / elapsed)
    print(f"use_cache={use_cache!s:5} -> {args.tokens / elapsed:7.0f} tokens/sec")

same = results[True][0][0, : model.cfg.block_size].tolist() == results[False][0][0, : model.cfg.block_size].tolist()
print(f"speed-up: {results[True][1] / results[False][1]:.1f}x | identical within the first context window: {same}")
