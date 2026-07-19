import torch

from frankengpt.config import GPTConfig
from frankengpt.model import GPT
from frankengpt.tokenizer import CharTokenizer
from frankengpt.training import TrainOptions, _scheduler, load_checkpoint, save_checkpoint


def test_checkpoint_loads_and_generation_extends_prompt(tmp_path):
    tokenizer = CharTokenizer.from_text("abc ")
    model = GPT(
        GPTConfig(
            tokenizer.vocab_size, context_length=8, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
    )
    optimizer = model.configure_optimizer(1e-3, 0.0)
    options = TrainOptions(max_steps=2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, tokenizer, optimizer, _scheduler(optimizer, options), 1, [], options
    )
    restored, restored_tokenizer, state = load_checkpoint(path, torch.device("cpu"))
    prompt = torch.tensor([restored_tokenizer.encode("ab")])
    output = restored.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=2)
    assert state["step"] == 1
    assert restored_tokenizer.tokens == tokenizer.tokens
    assert output.shape == (1, 5)
