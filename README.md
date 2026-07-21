# FrankenGPT: a trainable GPT from scratch

This repository continues the book-style journey from tokenization and embeddings to a
compact, end-to-end decoder-only Transformer. It trains locally on Mary Shelley's
*Frankenstein*, runs on CPU, and automatically uses CUDA mixed precision when a CUDA
PyTorch build is available.

## Learning path

Follow the steps in order. Steps 1–7 preserve the original learning notes and their
small, inspectable examples. Steps 8–12 continue from those foundations into a runnable
GPT training, checkpointing, and inference workflow. The maintained implementation lives
in `src/frankengpt`; `frankenlex_bootstrap.py` remains a bridge to the CLI for readers
following the earlier material.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The project uses the standard `pyproject.toml` editable-install workflow (PEP 660), so use
a current version of `pip` rather than invoking `setup.py` directly. A virtual environment
keeps the project dependencies separate from the system Python installation.

## Step 8: Build the decoder-only GPT

The token and positional embeddings from Step 7 feed a stack of pre-layer-norm Transformer
blocks. Each block applies causal multi-head self-attention, a residual connection, a
GELU MLP, and another residual connection. The final language-model head shares its
weights with the token embedding table. This is the next-token prediction model trained
by the following steps.

## Step 9: Train, resume, and sample

The included `data/pg84.txt` is the Gutenberg corpus. Start with this deliberately small
word-token experiment. It is quick enough for CPU and makes every training signal easy to
inspect. The output directory is ignored by Git.

```powershell
frankengpt train --data data/pg84.txt --tokenizer word --max-vocab 2048 `
  --device cpu --output runs/readme-smoke --max-steps 10 --batch-size 8 `
  --context-length 16 --d-model 32 --n-heads 4 --n-layers 2 --eval-interval 5
```

For a larger character-level scratch run, use a new output directory and at least
`--max-steps 2000`. Small runs validate the pipeline; they do not produce coherent prose.

### Verify the training result

The following is an **actual** CPU result from the command above. The loss is measured on
the training split and a held-out validation split. Both decrease, which shows the
optimizer is updating the model and improves beyond the current minibatch.

```text
{
  "device": "cpu",
  "parameters": 91520,
  "vocab_size": 2048,
  "tokens_per_second": 2737.4,
  "history": [
    {"step": 5.0, "train_loss": 7.6221, "val_loss": 7.6297, "lr": 0.00018},
    {"step": 10.0, "train_loss": 7.5760, "val_loss": 7.5936, "lr": 0.00030}
  ]
}
```

At this scale, use the loss history and checkpoints—not prose fluency—as the success
criteria. For a user-facing demonstration, use the optional pretrained workflow below and
fine-tune for substantially more than the smoke-test budget.

Throughput is machine-dependent; the parameter count, vocabulary size, and loss history
are the stable checks for this configuration.

Confirm that the run wrote `checkpoint_best.pt`, `checkpoint_last.pt`, and `metrics.json`:

```powershell
Get-ChildItem runs/readme-smoke
Get-Content runs/readme-smoke/metrics.json
```

## Step 10: Resume instead of starting over

The checkpoint stores the model, tokenizer, optimizer, scheduler, training history, and
current step. Continue the same run by increasing `--max-steps`; do not change the model
or tokenizer options when resuming.

```powershell
frankengpt train --data data/pg84.txt --tokenizer word --max-vocab 2048 `
  --device cpu --output runs/readme-smoke --resume runs/readme-smoke/checkpoint_last.pt `
  --max-steps 12 --batch-size 8 --context-length 16 --d-model 32 --n-heads 4 `
  --n-layers 2 --eval-interval 5
```

The resulting `metrics.json` keeps the earlier measurements and adds an evaluation at
step 12. This verifies that training resumed from saved state instead of creating a new
model.

## Step 11: Generate and judge the result

Load the best validation checkpoint and sample it. Temperature controls randomness;
`top-k` restricts each choice to the most likely tokens. Try the commands below to see
how prompt length and sampling affect the continuation.

```powershell
frankengpt generate --checkpoint runs/readme-smoke/checkpoint_best.pt --prompt "The" --max-new-tokens 24 --temperature 0.7 --top-k 10
frankengpt generate --checkpoint runs/readme-smoke/checkpoint_best.pt --prompt "I had worked hard" --max-new-tokens 24 --temperature 0.5 --top-k 5
frankengpt generate --checkpoint runs/readme-smoke/checkpoint_best.pt --prompt "I had worked hard for nearly two years" --max-new-tokens 24 --temperature 1.0 --top-k 30
```

One observed sample from the first command was:

```text
Prompt: The
Output: The and accents ship God, direction considered events soon disappeared, figure
disappeared disappeared books direction direction quickly the remained grief quickly
disappeared grief
```

This is **correct evidence of a trained and sampled model**, but it is not fluent prose:
ten updates on one small book are intentionally insufficient for coherence. The evidence
to look for is decreasing held-out loss, saved checkpoints, a successful resume, and
generation that returns learned corpus tokens rather than an error or random IDs.

On a CUDA-capable system use `--device cuda`; mixed precision is enabled automatically.
`--compile` opts in to `torch.compile` where PyTorch supports it. If the corpus is absent,
add `--download` to train.

## Step 12: Improve the showcase with more data

More steps on one book lead to overfitting. Download the curated public-domain classics
collection, then start a **new** word-token run; do not resume a single-book checkpoint
because its vocabulary is different.

```powershell
frankengpt fetch-data --output-dir data/classics
frankengpt train --data data/classics/*.txt --tokenizer word --max-vocab 16384 --device cuda --max-steps 10000 --batch-size 32 --output runs/classics-word
frankengpt generate --checkpoint runs/classics-word/checkpoint_best.pt --prompt "I had worked hard for nearly two years" --temperature 0.6 --top-k 20
```

### Optional polished local showcase: pretrained base

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

## Steps 1-7: original learning notes (preserved background)

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
