"""Train a compact GPT model on a UTF-8 text corpus."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import tiktoken
import torch
from torch.utils.data import DataLoader, random_split

from llm import (
    GPTConfig,
    GPTModel,
    TokenDataset,
    TrainConfig,
    cosine_learning_rate,
    estimate_loss,
    save_checkpoint,
    seed_everything,
    select_device,
)

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"


def load_corpus(path: Path, download_if_missing: bool) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    if not download_if_missing:
        raise FileNotFoundError(f"corpus not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(GUTENBERG_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/pg84.txt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/frankenstein.pt"))
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    seed_everything(train_config.seed)
    device = select_device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(load_corpus(args.data, args.download))
    dataset = TokenDataset(token_ids, args.context_length, args.stride)
    validation_size = max(1, int(0.1 * len(dataset)))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("corpus is too small for a train/validation split")
    training_data, validation_data = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)

    model_config = GPTConfig(
        vocab_size=tokenizer.n_vocab,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    model = GPTModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    start_step = 0
    if args.resume and args.output.exists():
        checkpoint = torch.load(args.output, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model_config.__dict__:
            raise ValueError("checkpoint model configuration does not match CLI configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1

    total_steps = args.epochs * len(train_loader)
    print(f"device={device} tokens={len(token_ids):,} examples={len(dataset):,}")
    print(f"parameters={model.parameter_count():,} total_steps={total_steps:,}")
    model.train()
    global_step = start_step
    for epoch in range(args.epochs):
        for x, y in train_loader:
            learning_rate = cosine_learning_rate(
                global_step, total_steps, train_config.learning_rate, train_config.warmup_steps
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x.to(device), y.to(device))
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            if global_step % train_config.eval_interval == 0:
                validation_loss = estimate_loss(model, validation_loader, device, train_config.eval_batches)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"train_loss={loss.item():.4f} val_loss={validation_loss:.4f} lr={learning_rate:.2e}"
                )
                save_checkpoint(args.output, model, optimizer, global_step, model_config, train_config)
            global_step += 1
    save_checkpoint(args.output, model, optimizer, global_step - 1, model_config, train_config)
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()
