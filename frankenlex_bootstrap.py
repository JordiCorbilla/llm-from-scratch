"""First-pass tokenizer bootstrap for LLM-from-scratch practice.

This script:
1) Downloads Frankenstein text from Project Gutenberg.
2) Prints total character count and the first 99 characters.
3) Performs a simple regex-based split into words and punctuation tokens.
4) Prints the first 30 tokens.
5) Builds a sorted unique vocabulary and prints vocab size.
6) Prints the first 51 vocabulary entries.
7) Adds special context tokens.
8) Builds a tokenizer class with encode/decode.
"""

from urllib.request import urlopen
import re
import sys


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
TOKEN_PATTERN = r"([,.:;?_!\"()\[\]'`]|--|\s)"


def download_text(url: str) -> str:
    """Download UTF-8 text from a URL."""
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def normalize_text(text: str) -> str:
    """Normalize punctuation to ASCII tokenizer-friendly forms."""
    return (
        text.replace("\ufeff", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "--")
    )


class SimpleTokenizerV2:
    """Regex-split tokenizer with special-token fallback for unknown words."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        text = normalize_text(text)
        preprocessed = re.split(TOKEN_PATTERN, text)
        preprocessed = [item.strip() for item in preprocessed if item and item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[token_id] for token_id in ids)
        # Remove spaces before punctuation.
        text = re.sub(r"\s+([.:;?!\"()'])", r"\1", text)
        # Merge apostrophe-based contractions back together: It ' s -> It's
        text = re.sub(r"'\s+(\w)", r"'\1", text)
        return text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw_text = normalize_text(download_text(GUTENBERG_URL))
    print(f"Total number of characters: {len(raw_text)}")
    print(raw_text[:99])

    preprocessed = re.split(TOKEN_PATTERN, raw_text)
    preprocessed = [item.strip() for item in preprocessed if item and item.strip()]
    print(preprocessed[:30])

    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|unk|>", "<|endoftext|>"])

    vocab_size = len(all_tokens)
    print(vocab_size)
    print(all_tokens[:51])

    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    tokenizer = SimpleTokenizerV2(vocab)

    text = "It's the last he did"
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    text_pair = "The modern prometheus . <|endoftext|> The Project Gutenberg ."
    pair_ids = tokenizer.encode(text_pair)
    print(pair_ids)
    print(tokenizer.decode(pair_ids))


if __name__ == "__main__":
    main()
