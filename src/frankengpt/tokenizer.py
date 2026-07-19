"""A deterministic character tokenizer suited to small local corpora."""

from __future__ import annotations

import json
import re
from collections import Counter
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


class WordTokenizer(CharTokenizer):
    """A compact regex word tokenizer for more readable local model samples."""

    unknown_token = "<unk>"
    pattern = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)

    @classmethod
    def from_text(cls, text: str, max_vocab: int = 2_048) -> WordTokenizer:
        if max_vocab < 2:
            raise ValueError("max_vocab must be at least 2")
        frequencies = Counter(cls.pattern.findall(text))
        ordered = sorted(frequencies, key=lambda token: (-frequencies[token], token))
        return cls([cls.unknown_token, *ordered[: max_vocab - 1]])

    def encode(self, text: str) -> list[int]:
        return [
            self.stoi.get(token, self.stoi[self.unknown_token])
            for token in self.pattern.findall(text)
        ]

    @property
    def forbidden_generation_ids(self) -> list[int]:
        """Token IDs that are useful for encoding but should never be sampled."""
        return [self.stoi[self.unknown_token]]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.itos.get(index, self.unknown_token) for index in ids)
        text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
        text = re.sub(r"([\(\[\{])\s+", r"\1", text)
        return re.sub(r"\s+(['’])\s+", r"\1", text)
