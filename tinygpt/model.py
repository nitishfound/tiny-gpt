"""A small GPT (decoder-only transformer) written from scratch in PyTorch.

Shape legend used in comments:
  B = batch size, T = sequence length (new tokens), S = total keys (cache + new),
  C = embedding size (n_embd), H = number of heads, D = head size (C // H), V = vocab size
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128  # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token can only look at itself and earlier tokens."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)  # Q, K, V in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular matrix: mask[i, j] is True when token i may look at token j (j <= i).
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, D): every head works on its own D-sized slice.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:  # KV cache: reuse keys/values of earlier tokens
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        S = k.size(2)

        # Scaled dot-product attention, written out by hand:
        # scores = Q K^T / sqrt(D)  -> mask the future -> softmax -> weighted sum of V
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, S)
        visible = self.mask[:, :, S - T : S, :S]  # the new queries sit at positions S-T .. S-1
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v  # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads back together
        return self.resid_dropout(self.proj(y)), (k, v)


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer block, pre-norm style: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past_kv=None):
        attn_out, present = self.attn(self.ln1(x), past_kv)
        x = x + attn_out  # residual connection: "communicate" between tokens
        x = x + self.mlp(self.ln2(x))  # residual connection: "think" per token
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # what the token is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # where the token is
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: same matrix in and out

        self.apply(self._init_weights)
        # GPT-2 trick: scale down the layers that write into the residual stream,
        # so the residual's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # The tied lm_head shares tok_emb's weights, so parameters() counts them once.
        return sum(p.numel() for p in self.parameters())

    def _run(self, idx: torch.Tensor, past_kvs: KVCache | None = None):
        B, T = idx.shape
        past_len = past_kvs[0][0].size(2) if past_kvs else 0
        if past_len + T > self.cfg.block_size:
            raise ValueError(f"sequence of {past_len + T} tokens is longer than block_size {self.cfg.block_size}")
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        presents: KVCache = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, past_kvs[i] if past_kvs else None)
            presents.append(present)
        logits = self.lm_head(self.ln_f(x))  # (B, T, V): a score for every possible next token
        return logits, presents

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits, _ = self._run(idx)
        loss = None
        if targets is not None:
            # Next-token prediction: compare the prediction at every position with the real next token.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation: predict one token, append it, repeat."""
        block = self.cfg.block_size
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self._run(idx[:, -block:])  # recompute the whole window every step
                nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
                idx = torch.cat([idx, nxt], dim=1)
            return idx

        # With a KV cache, each step only runs the ONE new token through the model.
        logits, cache = self._run(idx[:, -block:])
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat([idx, nxt], dim=1)
            if cache[0][0].size(2) >= block:
                # Window is full and positions are absolute: rebuild the cache
                # from the most recent half window, then keep going cheaply.
                logits, cache = self._run(idx[:, -(block // 2) :])
            else:
                logits, cache = self._run(nxt, cache)
        return idx


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id from logits of shape (B, V). Returns (B, 1)."""
    if temperature <= 0:  # greedy: always the most likely token
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature  # <1 sharpens the distribution, >1 flattens it
    if top_k is not None:  # keep only the k most likely tokens
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:  # nucleus: smallest set whose probability >= top_p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > top_p  # the top token is always kept
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
