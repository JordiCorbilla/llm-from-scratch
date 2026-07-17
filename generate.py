"""Generate text from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from llm import generate, load_model_checkpoint, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/frankenstein.pt"))
    parser.add_argument("--prompt", default="It was on a dreary night of November")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")
    model, _ = load_model_checkpoint(args.checkpoint, device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = generate(
        model,
        prompt_ids,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eot_token,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
