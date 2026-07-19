# FrankenGPT: a trainable GPT from scratch

This repository contains a compact, end-to-end decoder-only Transformer that trains locally on Mary Shelley's *Frankenstein*. It is intentionally small enough to run on CPU, but uses CUDA mixed precision automatically when a CUDA PyTorch build is available.

## Install

```powershell
python -m pip install -e ".[dev]"
```

## Train, resume, and sample

The included `data/pg84.txt` is the Gutenberg corpus. Use a short run to validate the
pipeline, then a longer run for the included showcase checkpoint:

```powershell
frankengpt train --device cpu --max-steps 100 --output runs/frankenstein
frankengpt train --device cpu --max-steps 150 --resume runs/frankenstein/checkpoint_last.pt --output runs/frankenstein
frankengpt generate --checkpoint runs/frankenstein/checkpoint_best.pt --prompt "I had worked hard" --temperature 0.7 --top-k 10
```

For the bundled character-level model, 100 steps only verifies that training works. It
does not produce coherent prose. The checked-in showcase checkpoint was trained through
2,000 steps; it is still a compact educational model rather than a production text model.

On a CUDA-capable system use `--device cuda`; mixed precision is enabled automatically. `--compile` opts in to `torch.compile` where PyTorch supports it. If the corpus is absent, add `--download` to train.

## Better showcase: train on multiple books

More steps on one book lead to overfitting. Download the curated public-domain classics
collection, then start a **new** word-token run; do not resume a single-book checkpoint
because its vocabulary is different.

```powershell
frankengpt fetch-data --output-dir data/classics
frankengpt train --data data/classics/*.txt --tokenizer word --max-vocab 16384 --device cuda --max-steps 10000 --batch-size 32 --output runs/classics-word
frankengpt generate --checkpoint runs/classics-word/checkpoint_best.pt --prompt "I had worked hard for nearly two years" --temperature 0.6 --top-k 20
```

## Polished local showcase (optional pretrained base)

The scratch models demonstrate the architecture but need far more data for fluent prose.
For a coherent demo, locally fine-tune `distilgpt2` on the same classics collection. This is
explicitly a pretrained-base workflow, distinct from the from-scratch model above.

```powershell
python -m pip install -e ".[showcase]"
frankengpt finetune-pretrained --data data/classics/*.txt --device cuda --max-steps 200 --output runs/distilgpt2-classics
frankengpt generate-pretrained --checkpoint runs/distilgpt2-classics --prompt "My dear Victor," --temperature 0.7 --top-k 30
```

## Architecture

- Character-level tokenizer fitted only from the training corpus (lossless and fast for the small local dataset).
- Optional frequency-limited word tokenizer (`--tokenizer word`) for a more readable local showcase.
- Learned token and positional embeddings.
- Pre-layer-norm decoder blocks: masked multi-head self-attention, residual connections, GELU MLP, and dropout.
- Tied input/output embedding weights, AdamW, warmup + cosine learning-rate decay, gradient clipping, checkpoints, and resumable optimizer/scheduler state.

The default model (`d_model=128`, 4 layers, 4 heads, 64-token context) has roughly 0.8M trainable parameters, depending on corpus vocabulary size.

## Commands

```text
frankengpt train [--help]      Train or resume and write checkpoint_{last,best}.pt plus metrics.json
frankengpt generate [--help]   Generate from a saved checkpoint; supports temperature and top-k sampling
frankengpt benchmark [--help]  Report forward/inference tokens per second and peak memory
```

## Validation

```powershell
ruff check .
pytest
frankengpt benchmark --checkpoint runs/frankenstein/checkpoint_best.pt --device cpu
```

## Original learning notes (preserved background)

The following notes document the original tokenizer and sliding-window work that led to
the application. They are retained as learning material and historical context; the
maintained, runnable implementation is under `src/frankengpt`, and
`frankenlex_bootstrap.py` now delegates to its CLI.

## Step 1: Load and cache training text

The script uses Project Gutenberg's Frankenstein text and caches it locally at
`data/pg84.txt` so it only downloads once.

## Step 2: Baseline regex tokenizer

The script builds a simple tokenizer with:

- vocabulary from sorted unique tokens
- special tokens `<|unk|>` and `<|endoftext|>`
- `SimpleTokenizerV2.encode(...)`
- `SimpleTokenizerV2.decode(...)`

This is a didactic tokenizer before switching to BPE.

## Step 3: Byte Pair Encoding with `tiktoken` (GPT-2)

The script demonstrates:

- `tiktoken.__version__`
- `tiktoken.get_encoding("gpt2")`
- encoding with `allowed_special={"<|endoftext|>"}`
- decoding IDs back to text
- unknown word decomposition (`"Akwirw ier"`) and reconstruction

## Step 4: Sliding-window input/target pairs

Given encoded tokens:

```python
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]
```

Create shifted pairs:

```python
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
```

`y` is always `x` shifted by one token.

## Step 5: Efficient PyTorch dataset + dataloader

Implemented:

- `GPTDatasetV1(Dataset)`
- `create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, suffle=True, drop_last=True, num_workers=0)`

`GPTDatasetV1` logic:

1. Encode full text once with GPT-2 BPE.
2. Slide a window over token IDs.
3. For each window:
   - `input_chunk = tokens[i : i + max_length]`
   - `target_chunk = tokens[i + 1 : i + max_length + 1]`
4. Return `(input_ids, target_ids)` tensors.

`create_dataloader_v1` creates a `DataLoader` over this dataset.

### Batch demo

The script includes:

```python
data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
```

and prints both to show sequential sliding behavior.

It also checks shift alignment:

```python
torch.equal(first_batch[0][:, 1:], first_batch[1][:, :-1])
```

Expected: `True`.

## Step 6: Max length vs stride tradeoff

The script prints stats for:

- `(max_length=2, stride=8)`
- `(max_length=8, stride=2)`
- `(max_length=8, stride=8)`
- `(max_length=8, stride=1)`

### Intuition diagram

Token positions:

```text
0 1 2 3 4 5 6 7 8 9 10 11 ...
```

Case A: `max_length=8, stride=8` (no overlap):

```text
[0........7] [8........15] [16.......23]
```

Case B: `max_length=8, stride=2` (heavy overlap):

```text
[0........7]
    [2........9]
        [4........11]
            [6........13]
```

Case C: `max_length=2, stride=8` (sparse coverage):

```text
[0.1]        [8.9]        [16.17]
```

### Practical guidance

- To avoid skipping tokens, use `stride <= max_length` and preferably `stride=1` for full coverage.
- Smaller stride increases overlap and training examples, but may increase overfitting risk due to repeated near-identical contexts.
- Larger stride reduces overlap and compute cost, but can skip many token positions.

## Step 7: Token + positional embeddings (256 dims)

Now convert token IDs into dense vectors for the model:

```python
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

Add positional information using context length:

```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
```

In the script, `context_length` is taken from a real batch shape and token embeddings
are computed from batch token IDs:

```python
token_embeddings = token_embedding_layer(batch_input_ids)
input_embeddings = token_embeddings + pos_embeddings
```

Expected shapes:

- `pos_embeddings`: `[context_length, 256]`
- `token_embeddings`: `[batch_size, context_length, 256]`
- `input_embeddings`: `[batch_size, context_length, 256]`

These `input_embeddings` are what the main LLM blocks (attention, MLP, etc.) consume.
