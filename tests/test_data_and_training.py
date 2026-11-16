import pytest
import torch

from tinygpt import CharTokenizer, get_batch, train_val_split
from tinygpt.checkpoint import load_checkpoint
from tinygpt.train import lr_at, parse_args, train

TEXT = "To be, or not to be, that is the question.\n" * 40


def test_char_tokenizer_round_trip():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode("to be")) == "to be"
    assert tok.vocab_size == len(set(TEXT))
    with pytest.raises(ValueError):
        tok.encode("Z")


def test_batches_are_shifted_by_one():
    tok = CharTokenizer.from_text(TEXT)
    train_data, val_data = train_val_split(tok.encode(TEXT), 0.1)
    assert len(train_data) + len(val_data) == len(TEXT)
    x, y = get_batch(train_data, 8, 4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_learning_rate_schedule():
    assert lr_at(0, 1.0, 0.1, warmup=10, total=100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 0.1, warmup=10, total=100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 0.1, warmup=10, total=100) == pytest.approx(0.1)
    middle = lr_at(55, 1.0, 0.1, warmup=10, total=100)
    assert 0.1 < middle < 1.0


def test_short_training_run_reduces_loss_and_saves_checkpoint(tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text(TEXT, encoding="utf-8")
    args = parse_args([
        "--data", str(data), "--out", str(tmp_path / "out"), "--steps", "120", "--batch-size", "8",
        "--block-size", "32", "--n-layer", "2", "--n-head", "2", "--n-embd", "32", "--dropout", "0",
        "--eval-interval", "60", "--eval-iters", "4", "--warmup", "10", "--device", "cpu",
    ])
    result = train(args)
    log = result["log"]
    assert log[-1]["train_loss"] < log[0]["train_loss"] * 0.6

    model, tok, meta = load_checkpoint(tmp_path / "out" / "ckpt.pt")
    assert meta["val_loss"] == pytest.approx(result["best_val"])
    out = model.generate(torch.tensor([tok.encode("To")]), 20, temperature=0)
    assert len(tok.decode(out[0].tolist())) == 22
