import torch

from llm import GPTConfig, GPTModel, TokenDataset, generate


def tiny_model() -> GPTModel:
    return GPTModel(
        GPTConfig(
            vocab_size=32,
            context_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
    )


def test_dataset_is_next_token_shifted() -> None:
    x, y = TokenDataset(list(range(20)), context_length=8, stride=4)[0]
    assert torch.equal(x[1:], y[:-1])


def test_forward_shape_and_loss() -> None:
    model = tiny_model()
    inputs = torch.randint(0, 32, (2, 8))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and torch.isfinite(loss)


def test_generation_adds_requested_tokens() -> None:
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3]])
    output = generate(model, prompt, max_new_tokens=4, temperature=0)
    assert output.shape == (1, 7)


def test_weight_tying() -> None:
    model = tiny_model()
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
