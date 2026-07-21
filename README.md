# FrankenGPT: train a GPT-style language model locally

FrankenGPT is a compact decoder-only Transformer that you can train, resume, benchmark,
and sample on your own machine. This guide follows the complete workflow from a fresh
checkout to a verified model checkpoint and generated text.

## Step 1 - Create an isolated Python environment

Use Python 3.10 or newer. In PowerShell, create and activate a virtual environment, then
install the project and its development tools.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm that the command-line application is available:

```powershell
frankengpt --help
```

## Step 2 - Understand what will be trained

The model learns next-token prediction. A sequence such as `The creature was` becomes an
input, and the desired target is the same sequence shifted by one token. The model is a
GPT-style decoder with:

- learned token and positional embeddings
- masked multi-head self-attention, so a token cannot see future tokens
- pre-layer normalization, residual connections, GELU MLPs, and dropout
- tied input/output embeddings
- AdamW, warmup plus cosine learning-rate decay, gradient clipping, checkpoints, and
  CUDA mixed precision when CUDA is selected

## Step 3 - Train a small model from scratch

Run this small CPU experiment first. `--download` obtains the Project Gutenberg
Frankenstein corpus when `data/pg84.txt` is not already present. The command uses a
word-level vocabulary and a deliberately small model so that it finishes quickly.

```powershell
frankengpt train --download --data data/pg84.txt --tokenizer word --max-vocab 2048 `
  --device cpu --output runs/readme-smoke --max-steps 10 --batch-size 8 `
  --context-length 16 --d-model 32 --n-heads 4 --n-layers 2 --eval-interval 5
```

The output directory contains all artifacts from the run and is ignored by Git.

## Step 4 - Verify that training actually happened

The command prints JSON with model size, throughput, and loss history. A verified CPU run
produced the following result. Throughput varies by machine; the parameter count,
vocabulary size, and loss trend are the useful checks.

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

Both the training loss and the held-out validation loss decreased. That is the first
evidence that optimization and validation are working. Inspect the files that were saved:

```powershell
Get-ChildItem runs/readme-smoke
Get-Content runs/readme-smoke/metrics.json
```

Expected files: `checkpoint_best.pt`, `checkpoint_last.pt`, and `metrics.json`.

## Step 5 - Resume from the saved checkpoint

Checkpoints contain the model, tokenizer, optimizer, scheduler, history, and current
step. Resume the same configuration with a larger `--max-steps` value:

```powershell
frankengpt train --data data/pg84.txt --tokenizer word --max-vocab 2048 `
  --device cpu --output runs/readme-smoke --resume runs/readme-smoke/checkpoint_last.pt `
  --max-steps 12 --batch-size 8 --context-length 16 --d-model 32 --n-heads 4 `
  --n-layers 2 --eval-interval 5
```

`metrics.json` should preserve the step 5 and step 10 records and add a new measurement
at step 12. Do not change model or tokenizer options when resuming a checkpoint.

## Step 6 - Generate text from the trained checkpoint

Load the best validation checkpoint and try different temperatures, top-k values, and
prompt lengths. Lower temperature and lower top-k make output more conservative; higher
values make it more varied.

```powershell
frankengpt generate --checkpoint runs/readme-smoke/checkpoint_best.pt --prompt "The" --max-new-tokens 24 --temperature 0.7 --top-k 10
frankengpt generate --checkpoint runs/readme-smoke/checkpoint_best.pt --prompt "I had worked hard" --max-new-tokens 24 --temperature 0.5 --top-k 5
frankengpt generate --checkpoint runs/readme-smoke/checkpoint_best.pt --prompt "I had worked hard for nearly two years" --max-new-tokens 24 --temperature 1.0 --top-k 30
```

One actual sample from the first command was:

```text
The and accents ship God, direction considered events soon disappeared, figure
disappeared disappeared books direction direction quickly the remained grief quickly
disappeared grief
```

The text is intentionally not fluent after only ten updates. The successful outcome here
is that the model trained, selected the best checkpoint by validation loss, reloaded it,
and generated tokens from its learned vocabulary without an error.

## Step 7 - Train a larger scratch model

For a more meaningful single-book experiment, use the character tokenizer and train for
at least 2,000 steps. Use a new output directory so the small smoke checkpoint remains
available for comparison.

```powershell
frankengpt train --download --device cpu --max-steps 2000 --output runs/frankenstein-char
frankengpt generate --checkpoint runs/frankenstein-char/checkpoint_best.pt `
  --prompt "I had worked hard" --temperature 0.7 --top-k 10
```

If CUDA is available, replace `--device cpu` with `--device cuda`. CUDA automatically
enables mixed precision. Add `--compile` to opt in to `torch.compile` when it is supported
by your PyTorch installation.

## Step 8 - Improve the data for a better showcase

Training longer on a single book eventually overfits. Download several public-domain
books and start a new word-token run. Do not resume a Frankenstein-only checkpoint because
the vocabulary is different.

```powershell
frankengpt fetch-data --output-dir data/classics
frankengpt train --data data/classics/*.txt --tokenizer word --max-vocab 16384 `
  --device cuda --max-steps 10000 --batch-size 32 --output runs/classics-word
frankengpt generate --checkpoint runs/classics-word/checkpoint_best.pt `
  --prompt "I had worked hard for nearly two years" --temperature 0.6 --top-k 20
```

If CUDA is unavailable, change the training command to `--device cpu` and reduce the
batch size if memory is limited.

## Step 9 - Benchmark the checkpoint

Benchmark reports forward-pass throughput, generation throughput, and peak memory use.

```powershell
frankengpt benchmark --checkpoint runs/readme-smoke/checkpoint_best.pt --device cpu
```

Run the same command with `--device cuda` on a CUDA machine to measure GPU performance.

## Step 10 - Run the project checks

Before changing the model, run the formatter checks and test suite:

```powershell
ruff check .
pytest
```

The tests cover tokenization, shifted dataset targets, Transformer forward passes, causal
attention masking, non-contiguous loss inputs, checkpoint loading, resumption safeguards,
and generation.

## Step 11 - Explore the available commands

Each command provides its own options and defaults:

```powershell
frankengpt train --help
frankengpt generate --help
frankengpt benchmark --help
frankengpt fetch-data --help
```

Use this sequence whenever you experiment: train, inspect the validation loss and saved
artifacts, resume if needed, generate with controlled sampling, and benchmark the result.
