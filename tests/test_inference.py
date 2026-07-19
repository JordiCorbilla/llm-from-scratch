import torch

from frankengpt.config import GPTConfig
from frankengpt.model import GPT


def test_inference_without_targets_returns_logits_and_no_loss():
    model = GPT(GPTConfig(vocab_size=11, context_length=8, d_model=16, n_heads=4, n_layers=1))
    logits, loss = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 11)
    assert loss is None
