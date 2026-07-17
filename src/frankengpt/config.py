"""Configuration objects for the model and training loop."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters for a decoder-only transformer."""

    vocab_size: int
    context_length: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    bias: bool = True

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)
