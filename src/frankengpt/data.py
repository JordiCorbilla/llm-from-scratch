"""Dataset and corpus-loading utilities."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"


def load_corpus(path: str | Path, download: bool = False) -> str:
    """Load UTF-8 text, optionally downloading Frankenstein if it is absent."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        if not download:
            raise FileNotFoundError(f"corpus not found: {corpus_path}")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(GUTENBERG_URL, timeout=30) as response:
            corpus_path.write_text(response.read().decode("utf-8"), encoding="utf-8")
    return corpus_path.read_text(encoding="utf-8").replace("\ufeff", "")


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Contiguous next-token prediction examples from a one-dimensional token stream."""

    def __init__(
        self, token_ids: list[int] | torch.Tensor, context_length: int, stride: int = 1
    ) -> None:
        if context_length < 1 or stride < 1:
            raise ValueError("context_length and stride must be positive")
        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.starts = range(0, max(0, len(self.tokens) - context_length), stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        return (
            self.tokens[start : start + self.context_length],
            self.tokens[start + 1 : start + self.context_length + 1],
        )
