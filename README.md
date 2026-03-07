# llm-from-scratch

Steps to build a large language model from scratch.

## Setup

Install dependency for byte-pair encoding:

```bash
python -m pip install tiktoken
```

Run the script:

```bash
python frankenlex_bootstrap.py
```

## Step 1: Read data and build a simple tokenizer

We start with Mary Shelley's *Frankenstein* from Project Gutenberg:

- Source URL: `https://www.gutenberg.org/cache/epub/84/pg84.txt`
- Script: `frankenlex_bootstrap.py`

The script:

1. Downloads raw text from Gutenberg.
2. Prints total number of characters with `len(raw_text)`.
3. Prints the first 99 characters with `raw_text[:99]`.
4. Tokenizes text using regex into words + punctuation.
5. Cleans tokens and removes empties.
6. Prints the first 30 tokens.

## Step 2: Build vocabulary and special tokens

Create deterministic vocab with unique sorted tokens and append special tokens.

```python
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])
vocab_size = len(all_tokens)
```

Then create mappings:

```python
vocab = {token: idx for idx, token in enumerate(all_tokens)}
```

## Step 3: Implement `SimpleTokenizerV2`

`SimpleTokenizerV2` provides:

- `encode(text)`: regex tokenize + map unknowns to `<|unk|>`.
- `decode(ids)`: map IDs back to tokens and clean spacing before punctuation.

Examples in script:

- Basic sentence encode/decode.
- Two sentences separated by `<|endoftext|>`.
- Unknown words triggering `<|unk|>`.

## Step 4: Byte Pair Encoding (BPE) with `tiktoken`

Use GPT-2 encoding:

```python
import tiktoken
print(tiktoken.__version__)
tokenizer = tiktoken.get_encoding("gpt2")
```

Encode text while allowing special token:

```python
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
```

Decode back:

```python
decoded = tokenizer.decode(integers)
```

### Unknown-word BPE example

Input:

```python
sample = "Akwirw ier"
ids = tokenizer.encode(sample)
```

Then inspect each token:

```python
for token_id in ids:
    print(token_id, tokenizer.decode([token_id]))
```

Finally check reconstruction:

```python
print(tokenizer.decode(ids) == sample)
```

With BPE, unknown words are split into known subword/byte units, so full decode should still reconstruct the original text.
