"""First-pass tokenizer bootstrap for LLM-from-scratch practice.

This script:
1) Downloads Frankenstein text from Project Gutenberg.
2) Prints total character count and the first 99 characters.
3) Performs a simple regex-based split into words and punctuation tokens.
4) Prints the first 30 tokens.
"""

from urllib.request import urlopen
import re
import sys


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"


def download_text(url: str) -> str:
    """Download UTF-8 text from a URL."""
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def main() -> None:
    # Avoid Windows console encoding errors when text contains BOM/Unicode chars.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw_text = download_text(GUTENBERG_URL)

    print(f"Total number of characters: {len(raw_text)}")
    print(raw_text[:99])

    # Split on punctuation + whitespace and keep punctuation as separate tokens.
    preprocessed = re.split(r"([,.:;?_!\"()\[\]'`]|--|\s)", raw_text)
    preprocessed = [item.strip() for item in preprocessed if item and item.strip()]

    print(preprocessed[:30])


if __name__ == "__main__":
    main()
