import torch

from frankengpt.config import GPTConfig
from frankengpt.model import GPT, CausalSelfAttention


def config() -> GPTConfig:
    return GPTConfig(
        vocab_size=17, context_length=8, d_model=16, n_heads=4, n_layers=2, dropout=0.0
    )


def test_transformer_forward_shape_loss_and_weight_tying():
    model = GPT(config())
    inputs = torch.randint(0, 17, (3, 6))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (3, 6, 17)
    assert loss is not None and torch.isfinite(loss)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_transformer_accepts_non_contiguous_targets():
    model = GPT(config())
    inputs = torch.randint(0, 17, (3, 6))
    targets = torch.randint(0, 17, (6, 3)).transpose(0, 1)
    _, loss = model(inputs, targets)
    assert loss is not None and torch.isfinite(loss)


def test_attention_never_assigns_probability_to_future_tokens():
    attention = CausalSelfAttention(config())
    _, weights = attention(torch.randn(1, 5, 16), return_attention=True)
    assert torch.allclose(
        weights[..., torch.triu(torch.ones(5, 5), diagonal=1).bool()], torch.zeros(1)
    )
