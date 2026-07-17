"""Training, checkpointing, evaluation, and throughput measurement."""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import GPTConfig
from .data import TokenDataset
from .model import GPT
from .tokenizer import CharTokenizer


@dataclass
class TrainOptions:
    max_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 10
    eval_interval: int = 25
    eval_batches: int = 10
    grad_clip: float = 1.0
    seed: int = 1337


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loaders(
    token_ids: list[int], context_length: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    split = max(context_length + 2, int(len(token_ids) * 0.9))
    train_data = TokenDataset(token_ids[:split], context_length)
    val_data = TokenDataset(token_ids[split - context_length :], context_length)
    if not train_data or not val_data:
        raise ValueError("corpus is too short for the selected context length")
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False),
    )


@torch.no_grad()
def estimate_loss(model: GPT, loader: DataLoader, device: torch.device, batches: int) -> float:
    model.eval()
    losses = []
    for index, (inputs, targets) in enumerate(loader):
        if index >= batches:
            break
        _, loss = model(inputs.to(device), targets.to(device))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def _scheduler(
    optimizer: torch.optim.Optimizer, options: TrainOptions
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < options.warmup_steps:
            return (step + 1) / max(1, options.warmup_steps)
        progress = (step - options.warmup_steps) / max(1, options.max_steps - options.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_checkpoint(
    path: str | Path,
    model: GPT,
    tokenizer: CharTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    history: list[dict[str, float]],
    options: TrainOptions,
) -> None:
    """Save all state required for deterministic continuation of a training run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "tokenizer": tokenizer.tokens,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "history": history,
            "train_options": asdict(options),
        },
        target,
    )


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[GPT, CharTokenizer, dict[str, Any]]:
    """Load model and tokenizer state from a checkpoint without restoring training state."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model, CharTokenizer(state["tokenizer"]), state


def train_model(
    text: str,
    config: GPTConfig,
    output_dir: str | Path,
    options: TrainOptions,
    device_name: str = "auto",
    resume: str | Path | None = None,
    compile_model: bool = False,
) -> dict[str, Any]:
    """Train or resume a GPT model and return recorded metrics."""
    torch.manual_seed(options.seed)
    device = select_device(device_name)
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_loader, val_loader = make_loaders(token_ids, config.context_length, options.batch_size)
    model = GPT(config).to(device)
    train_forward = model
    if compile_model and hasattr(torch, "compile"):
        # Keep the uncompiled module as the checkpoint source; compiled state dict keys differ.
        train_forward = torch.compile(model)
    optimizer = model.configure_optimizer(options.learning_rate, options.weight_decay)
    scheduler = _scheduler(optimizer, options)
    start_step, history = 0, []
    if resume:
        saved_model, saved_tokenizer, state = load_checkpoint(resume, device)
        if saved_model.config != config or saved_tokenizer.tokens != tokenizer.tokens:
            raise ValueError(
                "checkpoint model configuration or tokenizer does not match this training run"
            )
        model.load_state_dict(saved_model.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, history = int(state["step"]), list(state["history"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    out = Path(output_dir)
    best_loss = min((entry["val_loss"] for entry in history), default=float("inf"))
    started = time.perf_counter()
    for step in range(start_step, options.max_steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, loss = train_forward(inputs.to(device), targets.to(device))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if (step + 1) % options.eval_interval == 0 or step + 1 == options.max_steps:
            train_loss = estimate_loss(model, train_loader, device, options.eval_batches)
            val_loss = estimate_loss(model, val_loader, device, options.eval_batches)
            entry = {
                "step": float(step + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(entry)
            save_checkpoint(
                out / "checkpoint_last.pt",
                model,
                tokenizer,
                optimizer,
                scheduler,
                step + 1,
                history,
                options,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    out / "checkpoint_best.pt",
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    step + 1,
                    history,
                    options,
                )
    duration = time.perf_counter() - started
    stats = {
        "device": str(device),
        "parameters": model.parameter_count,
        "duration_seconds": duration,
        "tokens_per_second": (options.max_steps - start_step)
        * options.batch_size
        * config.context_length
        / duration,
        "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def benchmark(
    model: GPT, batch_size: int, device_name: str = "auto", iterations: int = 20
) -> dict[str, float | int | str]:
    """Measure training and generation throughput for a representative synthetic batch."""
    device = select_device(device_name)
    model = model.to(device).eval()
    inputs = torch.randint(
        model.config.vocab_size, (batch_size, model.config.context_length), device=device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tracemalloc.start()
    forward_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_elapsed = time.perf_counter() - forward_started
    prompt = inputs[:, :8]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    model.generate(prompt, max_new_tokens=16, temperature=1.0, top_k=10)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_elapsed = time.perf_counter() - generation_started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "device": str(device),
        "forward_tokens_per_second": batch_size
        * model.config.context_length
        * iterations
        / forward_elapsed,
        "inference_tokens_per_second": batch_size * 16 / generation_elapsed,
        "peak_memory_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else peak
        ),
    }
