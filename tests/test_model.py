import math

import pytest
import torch
import torch.nn.functional as F

from tinygpt import GPT, GPTConfig, sample_next

CFG = GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_output_shapes_and_loss(model):
    x = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss.ndim == 0


def test_initial_loss_is_close_to_uniform_guessing(model):
    # A fresh model should be unsure: loss close to ln(vocab_size).
    x = torch.randint(0, CFG.vocab_size, (8, 16))
    _, loss = model(x, torch.randint(0, CFG.vocab_size, (8, 16)))
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3


def test_causal_mask_future_tokens_do_not_change_the_past(model):
    x = torch.randint(0, CFG.vocab_size, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % CFG.vocab_size  # change only positions 8+
    logits_x, _ = model(x)
    logits_y, _ = model(y)
    assert torch.allclose(logits_x[0, :8], logits_y[0, :8], atol=1e-6)
    assert not torch.allclose(logits_x[0, 8:], logits_y[0, 8:])


def test_hand_written_attention_matches_pytorch_builtin(model):
    attn = model.blocks[0].attn
    x = torch.randn(2, 10, CFG.n_embd)
    ours, _ = attn(x)
    # Same computation using torch's fused kernel
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=2)
    q, k, v = (t.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.proj(y.transpose(1, 2).contiguous().view(B, T, C))
    assert torch.allclose(ours, ref, atol=1e-5)


def test_weight_tying(model):
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_rejects_sequences_longer_than_block_size(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, CFG.block_size + 1, dtype=torch.long))


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = model(x, y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2  # learned to memorise the batch


def test_kv_cache_gives_identical_greedy_output(model):
    prompt = torch.randint(0, CFG.vocab_size, (2, 4))
    fast = model.generate(prompt, 10, temperature=0, use_cache=True)
    slow = model.generate(prompt, 10, temperature=0, use_cache=False)
    assert torch.equal(fast, slow)


def test_generation_can_run_past_the_context_window(model):
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 50, temperature=1.0)
    assert out.shape == (1, 51)


def test_sampling_is_reproducible_with_a_seed(model):
    prompt = torch.zeros(1, 1, dtype=torch.long)
    a = model.generate(prompt, 20, generator=torch.Generator().manual_seed(7))
    b = model.generate(prompt, 20, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_top_k_and_top_p_only_pick_allowed_tokens():
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0, -1.0]])
    gen = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, 1.0, top_k=2, generator=gen).item() for _ in range(200)}
    assert picks <= {0, 1}
    # probs ~ [0.71, 0.26, 0.01, ...]: top_p=0.5 keeps only token 0
    picks = {sample_next(logits, 1.0, top_p=0.5, generator=gen).item() for _ in range(100)}
    assert picks == {0}
    assert sample_next(logits, temperature=0).item() == 0  # greedy
