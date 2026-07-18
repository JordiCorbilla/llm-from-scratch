"""Command-line interface for train, generate, and benchmark workflows."""

from __future__ import annotations

import argparse
import json

import torch

from .config import GPTConfig
from .data import CLASSIC_CORPORA, download_classics, load_corpora
from .training import TrainOptions, benchmark, load_checkpoint, select_device, train_model


def _config(args: argparse.Namespace, vocab_size: int) -> GPTConfig:
    return GPTConfig(
        vocab_size, args.context_length, args.d_model, args.n_heads, args.n_layers, args.dropout
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frankengpt", description="Train a small GPT on Frankenstein."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", nargs="+", default=["data/pg84.txt"])
    train.add_argument("--output", default="runs/frankenstein")
    train.add_argument("--download", action="store_true")
    train.add_argument("--resume")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train.add_argument("--max-steps", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--context-length", type=int, default=64)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--n-layers", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--eval-interval", type=int, default=25)
    train.add_argument("--compile", action="store_true")
    train.add_argument("--tokenizer", choices=["char", "word"], default="char")
    train.add_argument("--max-vocab", type=int, default=2_048)
    fetch = sub.add_parser("fetch-data", help="Download curated Project Gutenberg classics.")
    fetch.add_argument("--output-dir", default="data/classics")
    fetch.add_argument("--sources", nargs="*", choices=sorted(CLASSIC_CORPORA))
    generate = sub.add_parser("generate")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=160)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bench = sub.add_parser("benchmark")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--batch-size", type=int, default=8)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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
