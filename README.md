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
7. Creates a sorted list of unique tokens.
8. Adds special context tokens: `<|unk|>` and `<|endoftext|>`.
9. Prints vocabulary size.
10. Prints first 51 vocabulary entries for inspection.
11. Builds a `SimpleTokenizerV2` class with `encode` and `decode`.
12. Encodes and decodes a sample sentence.

### Why this matters

This gives us a basic token stream that can be mapped into IDs and later converted
into embeddings for LLM training.

### Run it

```bash
python frankenlex_bootstrap.py
```

### Current tokenizer approach

```python
preprocessed = re.split(r"([,.:;?_!\"()\\[\\]'`]|--|\\s)", raw_text)
preprocessed = [item.strip() for item in preprocessed if item and item.strip()]
```

This is intentionally simple and a good first pass before moving to a more robust
tokenizer (e.g., byte-pair encoding).

## Step 2: Build token IDs foundation (vocabulary)

Once tokens are created, we build the vocabulary by taking unique tokens and sorting
them alphabetically, then adding the special tokens.

```python
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])
vocab_size = len(all_tokens)
print(vocab_size)
print(all_tokens[:51])
```

Why this step:

- `set(preprocessed)` keeps only unique tokens.
- `sorted(...)` makes ordering deterministic.
- `all_tokens.extend(["<|unk|>", "<|endoftext|>"])` adds special context tokens.
- `vocab_size` tells us how many token IDs we need.
- The first 51 entries are printed for quick sanity checking before mapping tokens to IDs.

## Step 3: Implement `SimpleTokenizerV2` (encode/decode)

After creating the vocabulary, we map tokens to IDs and IDs back to tokens.

```python
vocab = {token: idx for idx, token in enumerate(all_tokens)}
tokenizer = SimpleTokenizerV2(vocab)
```

The class includes:

- `encode(self, text)`: split text with the same regex preprocessing and convert tokens to IDs.
- Unknown words are replaced with `<|unk|>` during encoding.
- `decode(self, ids)`: convert IDs back to text and remove spaces before punctuation with `r"\s+([.:;?!\"()'])"`.
- Unicode punctuation normalization (`\u2019`, `\u201c`, `\u2014`, BOM) keeps corpus and input tokenization consistent.

Demo in script:

```python
text = "It's the last he did"
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
```
