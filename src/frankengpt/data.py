"""Dataset and corpus-loading utilities."""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from torch.utils.data import Dataset

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
CLASSIC_CORPORA = {
    "frankenstein": GUTENBERG_URL,
    "alice": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "pride-prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sherlock-holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "dracula": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
}


def _download(url: str) -> str:
    request = Request(url, headers={"User-Agent": "frankengpt/0.1 educational project"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg catalogue headers and licence footers when present."""
    start = re.search(r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if start:
        text = text[start.end() :]
    end = re.search(r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if end:
        text = text[: end.start()]
    return text.strip()


def load_corpus(path: str | Path, download: bool = False) -> str:
    """Load UTF-8 text, optionally downloading Frankenstein if it is absent."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        if not download:
            raise FileNotFoundError(f"corpus not found: {corpus_path}")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text(_download(GUTENBERG_URL), encoding="utf-8")
    return strip_gutenberg_boilerplate(
        corpus_path.read_text(encoding="utf-8").replace("\ufeff", "")
    )


def load_corpora(paths: list[str | Path], download: bool = False) -> str:
    """Combine one or more local corpora with clear document boundaries."""
    if not paths:
        raise ValueError("at least one corpus path is required")
    expanded = [Path(match) for path in paths for match in (glob(str(path)) or [str(path)])]
    if download and len(expanded) > 1:
        raise ValueError("use fetch-data to download multiple curated corpora")
    return "\n\n\n\n".join(load_corpus(path, download=download) for path in expanded)


def download_classics(output_dir: str | Path, sources: list[str] | None = None) -> list[Path]:
    """Download the curated public-domain English classics collection."""
    selected = sources or list(CLASSIC_CORPORA)
    unknown = sorted(set(selected) - CLASSIC_CORPORA.keys())
    if unknown:
        raise ValueError(f"unknown corpus source(s): {', '.join(unknown)}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in selected:
        path = destination / f"{source}.txt"
        if not path.exists():
            path.write_text(_download(CLASSIC_CORPORA[source]), encoding="utf-8")
        paths.append(path)
    return paths


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
