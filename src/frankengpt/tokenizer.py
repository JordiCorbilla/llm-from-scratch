"""A deterministic character tokenizer suited to small local corpora."""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Losslessly encode corpus characters using a vocabulary built from training text."""

    def __init__(self, tokens: list[str]) -> None:
        if not tokens or len(tokens) != len(set(tokens)):
            raise ValueError("tokens must be a non-empty sequence of unique characters")
        self.tokens = tokens
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for index, token in enumerate(tokens)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"character not in tokenizer vocabulary: {error.args[0]!r}") from error

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[index] for index in ids)
        except KeyError as error:
            raise ValueError(f"token id outside vocabulary: {error.args[0]}") from error

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"tokens": self.tokens}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["tokens"])
