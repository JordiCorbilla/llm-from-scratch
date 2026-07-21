# How FrankenGPT works

This walkthrough connects the user-facing training command to the implementation in
`src/frankengpt`. Its goal is to make every stage of the from-scratch model traceable:
text becomes token IDs, token IDs become predictions, prediction errors update random
weights, and the learned weights generate new tokens.

## The complete path

```text
text
  -> corpus cleanup and train/validation split
  -> tokenizer vocabulary built from that text
  -> shifted input and target token windows
  -> token embeddings + position embeddings
  -> causal Transformer decoder blocks
  -> vocabulary logits
  -> cross-entropy next-token loss
  -> AdamW update
  -> checkpoint
  -> one-token-at-a-time generation
```

Nothing in this path loads pretrained model weights. The optional
`finetune-pretrained` command is a separate showcase workflow.

## 1. Load the corpus

`src/frankengpt/data.py` loads one or more UTF-8 text files. It can download
*Frankenstein* from Project Gutenberg and removes the catalogue header and licence footer
before training. Multiple books are joined with clear document boundaries.

The token stream is split contiguously: approximately 90% is used for training and 10%
for validation. Training batches update the model. Validation batches only measure how
well the learned patterns transfer to held-out text.

## 2. Build a tokenizer from the corpus

`src/frankengpt/tokenizer.py` provides two tokenizers:

| Tokenizer | A token represents | Best use in this project |
| --- | --- | --- |
| `char` | One character | Lossless, small vocabulary, longer meaningful runs. |
| `word` | A word or punctuation mark | Fast smoke tests with readable sampled tokens. |

Both vocabularies are fitted from the supplied corpus. For the character tokenizer,
sorted unique characters receive integer IDs. For the word tokenizer, the most frequent
words and punctuation receive IDs up to `--max-vocab`; unseen words map to `<unk>`.

Example:

```text
text:       The creature
tokens:     ["The", "creature"]
token IDs:  [418, 937]
```

The exact IDs depend on the training corpus. They have no meaning by themselves; they are
indexes into a vocabulary and, later, an embedding table.

## 3. Create next-token examples

`TokenDataset` in `src/frankengpt/data.py` creates fixed-length windows. Each target is the
input shifted one token to the left:

```text
token stream:  [12, 31, 7, 44, 9]
input:         [12, 31, 7, 44]
target:        [31, 7, 44, 9]
```

For a batch size `B` and context length `T`, both tensors have shape `[B, T]`. The model
must predict every target token using only the input tokens at or before that position.

## 4. Convert IDs into vectors

`GPT` in `src/frankengpt/model.py` owns two learned embedding tables:

- token embeddings say what each vocabulary item currently means to the model;
- position embeddings tell the model where each token occurs in the context window.

With vocabulary size `V` and model width `C`:

```text
input IDs                  [B, T]
token embeddings           [B, T, C]
position embeddings           [T, C]
combined representation    [B, T, C]
```

These vectors begin as random values. Training changes them along with every other model
parameter.

## 5. Apply causal self-attention

Each `CausalSelfAttention` layer projects the current representation into queries, keys,
and values. It splits the model width across `H` attention heads:

```text
queries, keys, values  [B, H, T, C/H]
attention scores       [B, H, T, T]
```

For each head, the layer:

1. compares queries with keys using scaled dot products;
2. replaces scores for future positions with negative infinity;
3. converts the remaining scores to probabilities with softmax;
4. uses those probabilities to combine value vectors; and
5. joins all heads and applies an output projection.

The triangular causal mask is essential. A token at position 3 may attend to positions
0–3, but never to position 4 or later. Without that rule, the model could see the answer
while training and would not learn valid autoregressive generation.

## 6. Build a decoder block

Each `Block` applies pre-layer normalization, causal attention, an MLP, and two residual
connections:

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

The MLP expands the width from `C` to `4C`, applies GELU, and projects back to `C`.
Residual connections give information and gradients a direct path through the network.
The configured number of blocks is controlled by `--n-layers`.

## 7. Predict the next token

After the final layer normalization, the language-model head maps each vector to one
score per vocabulary item:

```text
hidden states  [B, T, C]
logits         [B, T, V]
```

The output head shares weights with the token embedding table. This weight tying reduces
the parameter count and lets the same learned token representation support both input and
output.

Cross-entropy compares the logits with the shifted targets. A lower loss means the model
assigned more probability to the correct next tokens.

## 8. Update random weights

`src/frankengpt/training.py` uses:

- AdamW optimization;
- linear learning-rate warmup followed by cosine decay;
- weight decay on matrix parameters;
- gradient clipping for stability;
- CUDA autocast and gradient scaling when CUDA is selected; and
- periodic training and validation loss estimates.

One optimizer step is:

```text
forward pass -> loss -> backpropagation -> clip gradients -> AdamW update
```

The smoke run performs only ten updates. That proves the pipeline works, but it is far too
small a budget for fluent text.

## 9. Save and resume checkpoints

At each evaluation, training writes:

- `checkpoint_last.pt`: the newest model and optimizer state;
- `checkpoint_best.pt`: the state with the lowest validation loss so far; and
- `metrics.json`: device, parameter count, throughput, and loss history.

A checkpoint contains model configuration, tokenizer vocabulary, model weights, optimizer
state, scheduler state, training step, and history. Resume with the same corpus, tokenizer,
and model shape; increase `--max-steps` to continue.

Only load `.pt` checkpoints you trust because PyTorch checkpoints can contain pickled
Python data.

## 10. Generate autoregressively

Generation starts with the prompt token IDs. For every new token, `GPT.generate`:

1. crops a long history to the model's context length;
2. runs the model and selects the logits at the last position;
3. divides logits by temperature;
4. optionally keeps only the `top_k` most likely choices;
5. samples one token; and
6. appends it before repeating the process.

Lower temperature and smaller `top_k` make sampling more conservative. Higher values
increase variety and mistakes. Sampling cannot compensate for an undertrained model; loss,
data quality, and training scale matter first.

## Source map

| File | Responsibility |
| --- | --- |
| `src/frankengpt/tokenizer.py` | Character and word vocabularies; encode and decode. |
| `src/frankengpt/data.py` | Download, clean, combine, and window corpus text. |
| `src/frankengpt/config.py` | Model dimensions and hyperparameters. |
| `src/frankengpt/model.py` | Attention, MLP, decoder blocks, loss, and generation. |
| `src/frankengpt/training.py` | Loaders, optimization, evaluation, checkpoints, and benchmarks. |
| `src/frankengpt/cli.py` | User-facing commands and arguments. |
| `src/frankengpt/pretrained.py` | Separate optional Hugging Face showcase. |
| `tests/` | Executable checks for tokenization, masking, training, and checkpoints. |

## Experiments that teach the most

Change one variable at a time and compare validation loss and generated text:

| Experiment | Option | What it demonstrates |
| --- | --- | --- |
| More context | `--context-length` | How much history each prediction can use. |
| Wider representations | `--d-model` | Model capacity and compute cost. |
| More decoder blocks | `--n-layers` | Depth, capacity, and optimization cost. |
| Different token units | `--tokenizer char\|word` | Vocabulary size versus sequence length. |
| More data | `--data ...` | Generalization and reduced single-book overfitting. |
| More updates | `--max-steps` | Undertraining, learning progress, and eventual overfitting. |
| Different sampling | `--temperature`, `--top-k` | Randomness at inference without retraining. |

Return to the [README learning path](../README.md) to run the complete workflow.
