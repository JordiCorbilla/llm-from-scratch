# Frankenstein training report

## Configuration

The completed CPU run used the bundled Gutenberg text, a 90/10 contiguous split, a character vocabulary, 64-token contexts, 4 decoder blocks, 4 attention heads, `d_model=128`, dropout `0.1`, AdamW, warmup plus cosine decay, and gradient clipping. It contains **813,440** trainable parameters.

## Verified training and checkpoints

The run trained from scratch to step 100, resumed to 110, then resumed to 1,000 and 2,000 steps. `checkpoint_last.pt` and `checkpoint_best.pt` are produced at every evaluation; the best checkpoint is used for generation.

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 25 | 3.440 | 3.408 |
| 1,000 | 2.296 | 2.234 |
| 2,000 | 2.120 | 1.992 |

Loss decreased by 1.416 validation-loss points. The final 1,000-step CPU extension sustained 5,822 training tokens/sec. A CUDA mixed-precision smoke run and checkpoint save also completed successfully.

## Generation assessment

Sampling was verified with short and long prompts, temperatures `0.5`, `0.8`, and `1.1`, and top-k values `5`, `20`, and unrestricted. The model produces word-like English fragments, but not dependable prose. This is expected for a small character-level model trained on a single 456 KB novel. Scaling the context length, model width/depth, corpus, and optimization budget—preferably with a subword tokenizer—are the highest-impact improvements.

## Benchmarking

The `benchmark` command measures actual autoregressive generation tokens/sec separately from full-context forward tokens/sec. On CUDA it reports native PyTorch peak allocation; CPU's reported peak is Python allocation observed by `tracemalloc`, so it deliberately excludes allocator/native tensor memory.
