"""Command-line interface for train, generate, and benchmark workflows."""

from __future__ import annotations

import argparse
import json

import torch

from .config import GPTConfig
from .data import CLASSIC_CORPORA, download_classics, load_corpora
from .pretrained import finetune_pretrained, generate_pretrained
from .training import TrainOptions, benchmark, load_checkpoint, select_device, train_model


def _config(args: argparse.Namespace, vocab_size: int) -> GPTConfig:
    return GPTConfig(
        vocab_size, args.context_length, args.d_model, args.n_heads, args.n_layers, args.dropout
    )


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        prog="frankengpt",
        description="Train, evaluate, and sample a compact GPT-style language model.",
        formatter_class=formatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser(
        "train",
        help="Train from scratch or resume a compatible checkpoint.",
        description="Train from scratch or resume a compatible checkpoint.",
        formatter_class=formatter,
    )
    train.add_argument(
        "--data", nargs="+", default=["data/pg84.txt"], help="Corpus paths or glob patterns."
    )
    train.add_argument("--output", default="runs/frankenstein", help="Run artifact directory.")
    train.add_argument(
        "--download",
        action="store_true",
        help="Download Frankenstein when the default corpus path is missing.",
    )
    train.add_argument("--resume", help="Last checkpoint from a compatible training run.")
    train.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    train.add_argument("--max-steps", type=int, default=100, help="Total optimizer steps.")
    train.add_argument("--batch-size", type=int, default=32, help="Sequences per optimizer step.")
    train.add_argument(
        "--context-length", type=int, default=64, help="Tokens in each training sequence."
    )
    train.add_argument("--d-model", type=int, default=128, help="Embedding and hidden width.")
    train.add_argument("--n-heads", type=int, default=4, help="Attention heads per layer.")
    train.add_argument("--n-layers", type=int, default=4, help="Transformer decoder blocks.")
    train.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    train.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Peak AdamW learning rate."
    )
    train.add_argument(
        "--eval-interval", type=int, default=25, help="Steps between loss evaluations."
    )
    train.add_argument(
        "--compile", action="store_true", help="Use torch.compile when it is available."
    )
    train.add_argument(
        "--tokenizer", choices=["char", "word"], default="char", help="Tokenizer type."
    )
    train.add_argument(
        "--max-vocab", type=int, default=2_048, help="Maximum word-token vocabulary size."
    )
    fetch = sub.add_parser(
        "fetch-data",
        help="Download curated Project Gutenberg classics.",
        description="Download curated Project Gutenberg classics.",
        formatter_class=formatter,
    )
    fetch.add_argument("--output-dir", default="data/classics", help="Download directory.")
    fetch.add_argument(
        "--sources",
        nargs="*",
        choices=sorted(CLASSIC_CORPORA),
        help="Books to download; omit to download the complete collection.",
    )
    fine_tune = sub.add_parser(
        "finetune-pretrained",
        help="Fine-tune a Hugging Face causal language model.",
        description="Fine-tune a Hugging Face causal language model on local text.",
        formatter_class=formatter,
    )
    fine_tune.add_argument("--data", nargs="+", required=True, help="Corpus paths or globs.")
    fine_tune.add_argument(
        "--output", default="runs/distilgpt2-classics", help="Hugging Face output directory."
    )
    fine_tune.add_argument("--model", default="distilgpt2", help="Base model ID or local path.")
    fine_tune.add_argument("--max-steps", type=int, default=200, help="Optimizer steps.")
    fine_tune.add_argument("--batch-size", type=int, default=2, help="Token blocks per step.")
    fine_tune.add_argument("--block-size", type=int, default=128, help="Tokens per text block.")
    fine_tune.add_argument(
        "--learning-rate", type=float, default=5e-5, help="AdamW learning rate."
    )
    fine_tune.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    generate_pretrained_parser = sub.add_parser(
        "generate-pretrained",
        help="Generate text from a fine-tuned Hugging Face checkpoint.",
        description="Generate text from a fine-tuned Hugging Face checkpoint.",
        formatter_class=formatter,
    )
    generate_pretrained_parser.add_argument(
        "--checkpoint", required=True, help="Fine-tuned model directory."
    )
    generate_pretrained_parser.add_argument("--prompt", required=True, help="Prompt text.")
    generate_pretrained_parser.add_argument(
        "--max-new-tokens", type=int, default=160, help="Maximum generated tokens."
    )
    generate_pretrained_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    generate_pretrained_parser.add_argument(
        "--top-k", type=int, default=30, help="Limit sampling to the top-k tokens."
    )
    generate_pretrained_parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    generate = sub.add_parser(
        "generate",
        help="Generate text from a scratch-model checkpoint.",
        description="Generate text from a scratch-model checkpoint.",
        formatter_class=formatter,
    )
    generate.add_argument("--checkpoint", required=True, help="Trusted .pt checkpoint path.")
    generate.add_argument("--prompt", required=True, help="Prompt text in the model vocabulary.")
    generate.add_argument(
        "--max-new-tokens", type=int, default=160, help="Number of tokens to generate."
    )
    generate.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature."
    )
    generate.add_argument("--top-k", type=int, help="Limit sampling to the top-k tokens.")
    generate.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    bench = sub.add_parser(
        "benchmark",
        help="Measure forward and generation throughput.",
        description="Measure forward and autoregressive generation throughput.",
        formatter_class=formatter,
    )
    bench.add_argument("--checkpoint", required=True, help="Trusted .pt checkpoint path.")
    bench.add_argument("--batch-size", type=int, default=8, help="Synthetic batch size.")
    bench.add_argument(
        "--iterations", type=int, default=20, help="Timed forward-pass iterations."
    )
    bench.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        text = load_corpora(args.data, args.download)
        vocab_size = args.max_vocab if args.tokenizer == "word" else len(set(text))
        options = TrainOptions(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_interval=args.eval_interval,
        )
        print(
            json.dumps(
                train_model(
                    text,
                    _config(args, vocab_size),
                    args.output,
                    options,
                    args.device,
                    args.resume,
                    args.compile,
                    args.tokenizer,
                    args.max_vocab,
                ),
                indent=2,
            )
        )
    elif args.command == "fetch-data":
        paths = download_classics(args.output_dir, args.sources)
        print(json.dumps({"downloaded": [str(path) for path in paths]}, indent=2))
    elif args.command == "finetune-pretrained":
        print(
            json.dumps(
                finetune_pretrained(
                    load_corpora(args.data),
                    args.output,
                    args.model,
                    args.max_steps,
                    args.batch_size,
                    args.block_size,
                    args.learning_rate,
                    args.device,
                ),
                indent=2,
            )
        )
    elif args.command == "generate-pretrained":
        print(
            generate_pretrained(
                args.checkpoint,
                args.prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.device,
            )
        )
    elif args.command == "generate":
        device = select_device(args.device)
        model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
        model.eval()
        prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
        result = model.generate(
            prompt_ids,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            getattr(tokenizer, "forbidden_generation_ids", None),
        )[0].tolist()
        print(tokenizer.decode(result))
    else:
        device = select_device(args.device)
        model, _, _ = load_checkpoint(args.checkpoint, device)
        print(json.dumps(benchmark(model, args.batch_size, args.device, args.iterations), indent=2))


if __name__ == "__main__":
    main()
