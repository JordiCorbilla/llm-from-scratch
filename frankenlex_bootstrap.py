"""First-pass tokenizer bootstrap for LLM-from-scratch practice.

This script:
1) Downloads Frankenstein text from Project Gutenberg.
2) Prints total character count and the first 99 characters.
3) Performs a simple regex-based split into words and punctuation tokens.
4) Prints the first 30 tokens.
5) Builds a sorted unique vocabulary and prints vocab size.
6) Prints the first 51 vocabulary entries.
7) Builds a simple tokenizer class with encode/decode.
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
    """Normalize Unicode punctuation to simple ASCII tokenizer-friendly forms."""
    return (
        text.replace("\ufeff", "")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "--")
    )


class SimpleTokenizerV1:
    """Simple regex-split tokenizer with deterministic token-id mappings."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        text = normalize_text(text)
        preprocessed = re.split(TOKEN_PATTERN, text)
        preprocessed = [item.strip() for item in preprocessed if item and item.strip()]

        unknown_tokens = [token for token in preprocessed if token not in self.str_to_int]
        if unknown_tokens:
            raise ValueError(
                f"Unknown token(s) found: {unknown_tokens[:5]}. "
                "Expand vocab or add an <unk> token."
            )

        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[token_id] for token_id in ids)

        # Remove spaces before punctuation, and after opening brackets.
        text = re.sub(r"\s+([,.:;?_!\"()\[\]'`])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        # Merge apostrophe-based contractions back together: It ' s -> It's
        text = re.sub(r"'\s+(\w)", r"'\1", text)
        return text


def main() -> None:
    # Avoid Windows console encoding errors when text contains BOM/Unicode chars.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw_text = normalize_text(download_text(GUTENBERG_URL))

    print(f"Total number of characters: {len(raw_text)}")
    print(raw_text[:99])

    # Split on punctuation + whitespace and keep punctuation as separate tokens.
    preprocessed = re.split(TOKEN_PATTERN, raw_text)
    preprocessed = [item.strip() for item in preprocessed if item and item.strip()]

    print(preprocessed[:30])

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    print(all_words[:51])

    vocab = {token: idx for idx, token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV1(vocab)
    text = "It's the last he did"
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
