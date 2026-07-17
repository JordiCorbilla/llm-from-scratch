# LLM from scratch

A compact, fully trainable GPT-style language model implemented directly in PyTorch. The repository keeps the original tokenizer and embedding walkthrough, then completes the path through causal self-attention, transformer blocks, training, checkpointing, and text generation.

## What is included

- GPT-2 BPE tokenization with `tiktoken`
- Sliding-window next-token dataset
- Token and learned positional embeddings
- Multi-head causal self-attention using PyTorch scaled dot-product attention
- Pre-norm transformer blocks with GELU feed-forward networks
- Weight-tied language-model output head
- Cross-entropy next-token objective
- AdamW, gradient clipping, warmup, and cosine learning-rate decay
- Train/validation split and periodic validation loss
- Resumable checkpoints
- Greedy or temperature/top-k generation
- Automatic CUDA, Apple MPS, or CPU selection
- Unit tests and GitHub Actions CI

## Scope and expectations

This project produces a real autoregressive language model, but the default corpus is only *Frankenstein*. It is therefore a learning model and domain text generator, not a general-purpose assistant. Model quality depends primarily on corpus size, diversity, parameter count, context length, and training compute.

The default configuration is approximately a small GPT model suitable for experimentation on a modern GPU. For a quick CPU smoke test, use fewer layers, a smaller embedding dimension, and a shorter context.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Train the model

The trainer downloads and caches Project Gutenberg's *Frankenstein* at `data/pg84.txt` when the file is missing.

```bash
python train.py
```

Default output:

```text
checkpoints/frankenstein.pt
```

A practical GPU run:

```bash
python train.py \
  --context-length 256 \
  --embedding-dim 384 \
  --heads 6 \
  --layers 8 \
  --batch-size 16 \
  --epochs 20 \
  --stride 128
```

A fast CPU smoke test:

```bash
python train.py \
  --context-length 64 \
  --embedding-dim 128 \
  --heads 4 \
  --layers 2 \
  --batch-size 4 \
  --epochs 1 \
  --stride 64 \
  --device cpu
```

Resume an interrupted run using the same architecture flags:

```bash
python train.py --resume
```

Train on another UTF-8 corpus:

```bash
python train.py --data data/my_corpus.txt --no-download --output checkpoints/my_model.pt
```

## Generate text

```bash
python generate.py \
  --checkpoint checkpoints/frankenstein.pt \
  --prompt "It was on a dreary night of November" \
  --tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

Use `--temperature 0` for greedy deterministic decoding. Lower non-zero temperatures are more conservative; higher values increase variety and error rate.

## Architecture

For input token IDs of shape `[batch, sequence]`:

1. Token embeddings and learned positional embeddings are added.
2. Each pre-norm transformer block applies causal self-attention and an MLP through residual connections.
3. Final layer normalization feeds a vocabulary projection.
4. The output projection shares weights with the token embedding table.
5. Cross-entropy compares each position with the next token.

The causal mask prevents a token from attending to future tokens.

## Main files

| File | Purpose |
|---|---|
| `llm.py` | Model, dataset, generation, device, schedule, evaluation, and checkpoint utilities |
| `train.py` | End-to-end training CLI |
| `generate.py` | Checkpoint loading and text generation CLI |
| `frankenlex_bootstrap.py` | Original educational tokenizer, dataloader, and embedding walkthrough |
| `tests/test_llm.py` | Core behavioral tests |

## Test and lint

```bash
pytest -q
ruff check .
```

## Checkpoint format

Each `.pt` checkpoint contains:

- model state
- optimizer state
- global training step
- model configuration
- training configuration

Checkpoints and downloaded corpora are intentionally excluded from Git because they can be large and are reproducible.

## Scaling beyond the demo

The highest-value next improvements are:

1. Use a much larger, licensed, deduplicated corpus with held-out evaluation data.
2. Stream tokenized shards instead of holding the complete token list in memory.
3. Add mixed precision, gradient accumulation, distributed data parallelism, and compilation.
4. Add validation perplexity, generated-sample tracking, and experiment logging.
5. Replace learned absolute positions with RoPE for longer-context experiments.
6. Add instruction fine-tuning only after a competent base model exists.

Training a genuinely useful general-purpose LLM from random initialization requires orders of magnitude more text and compute than this repository's default example. For practical applications, adapting a strong open-weight pretrained model is normally more effective; this project is designed to make every core mechanism inspectable.

## Original learning walkthrough

Run the original staged tokenizer and embedding demonstration with:

```bash
python frankenlex_bootstrap.py
```
