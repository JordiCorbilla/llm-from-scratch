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
7. Creates a sorted list of unique tokens (vocabulary).
8. Prints vocabulary size.
9. Prints first 51 vocabulary entries for inspection.
10. Builds a `SimpleTokenizerV1` class with `encode` and `decode`.
11. Encodes and decodes a sample sentence.

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

## Step 2: Build token IDs foundation (vocabulary)

Once tokens are created, we build the vocabulary by taking unique tokens and sorting
them alphabetically.

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
print(all_words[:51])
```

Why this step:

- `set(preprocessed)` keeps only unique tokens.
- `sorted(...)` makes ordering deterministic.
- `vocab_size` tells us how many token IDs we need.
- The first 51 entries are printed for quick sanity checking before mapping tokens to IDs.

## Step 3: Implement `SimpleTokenizerV1` (encode/decode)

After creating the vocabulary, we map tokens to IDs and IDs back to tokens.

```python
vocab = {token: idx for idx, token in enumerate(all_words)}
tokenizer = SimpleTokenizerV1(vocab)
```

The class includes:

- `encode(self, text)`: split text with the same regex preprocessing and convert tokens to IDs.
- `decode(self, ids)`: convert IDs back to text and clean spacing around punctuation to reconstruct readable text.
- Unicode punctuation normalization (`’`, `“`, `—`, BOM) is applied to keep corpus and input tokenization consistent.

Demo in script:

```python
text = "It's the last he did"
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
```
