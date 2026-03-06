# llm-from-scratch

Steps to build a large language model from scratch.

## Step 1: Read data and build a simple tokenizer

We start with Mary Shelley's *Frankenstein* from Project Gutenberg:

- Source URL: `https://www.gutenberg.org/cache/epub/84/pg84.txt`
- Script: `frankenlex_bootstrap.py`

### What this script does

1. Downloads raw text from Gutenberg.
2. Prints total number of characters with `len(raw_text)`.
3. Prints the first 99 characters with `raw_text[:99]`.
4. Tokenizes the text with a simple regex split into words + special characters.
5. Cleans tokens using `.strip()` and removes empty values.
6. Prints the first 30 tokens.

### Why this matters

This gives us a basic token stream that can be mapped into IDs and later converted
into embeddings for LLM training.

### Run it

```bash
python frankenlex_bootstrap.py
```

### Current tokenizer approach

The script uses:

```python
preprocessed = re.split(r"([,.:;?_!\"()\\[\\]'`]|--|\\s)", raw_text)
preprocessed = [item.strip() for item in preprocessed if item and item.strip()]
```

This is intentionally simple and a good first pass before moving to a more robust
tokenizer (e.g., byte-pair encoding).
