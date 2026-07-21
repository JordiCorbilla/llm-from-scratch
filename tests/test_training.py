import pytest
import torch
from torch.utils.data import DataLoader

from frankengpt.config import GPTConfig
from frankengpt.model import GPT
from frankengpt.training import TrainOptions, estimate_loss, train_model


def test_estimate_loss_rejects_empty_loader():
    model = GPT(GPTConfig(vocab_size=8, context_length=2, d_model=8, n_heads=2, n_layers=1))
    with pytest.raises(ValueError, match="produced no batches"):
        estimate_loss(model, DataLoader([]), torch.device("cpu"), batches=1)


def test_word_training_uses_actual_tokenizer_vocabulary(tmp_path):
    config = GPTConfig(
        vocab_size=64, context_length=2, d_model=8, n_heads=2, n_layers=1, dropout=0.0
    )
    options = TrainOptions(max_steps=1, batch_size=1, eval_interval=1, eval_batches=1)
    stats = train_model(
        "one two. one two. one two. one two.",
        config,
        tmp_path,
        options,
        device_name="cpu",
        tokenizer_kind="word",
        max_vocab=64,
    )
    assert stats["vocab_size"] == 4
