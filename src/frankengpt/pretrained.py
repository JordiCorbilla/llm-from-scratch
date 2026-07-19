"""Optional local fine-tuning workflow for a coherent showcase model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .training import select_device


class HFTokenDataset(Dataset[torch.Tensor]):
    """Contiguous fixed-length blocks for Hugging Face causal-language-model training."""

    def __init__(self, ids: torch.Tensor, block_size: int) -> None:
        self.ids = ids
        self.block_size = block_size
        self.starts = range(0, max(0, len(ids) - block_size), block_size)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = self.starts[index]
        return self.ids[start : start + self.block_size]


def _transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the showcase extra first: python -m pip install -e '.[showcase]'"
        ) from error
    return AutoModelForCausalLM, AutoTokenizer


def finetune_pretrained(
    text: str,
    output_dir: str | Path,
    model_name: str = "distilgpt2",
    max_steps: int = 200,
    batch_size: int = 2,
    block_size: int = 128,
    learning_rate: float = 5e-5,
    device_name: str = "auto",
) -> dict[str, float | int | str]:
    """Fine-tune a pretrained causal LM locally and save it in Hugging Face format."""
    if max_steps < 1 or batch_size < 1 or block_size < 8:
        raise ValueError("max_steps, batch_size, and block_size must be positive")
    auto_model, auto_tokenizer = _transformers()
    device = select_device(device_name)
    tokenizer = auto_tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1_000_000_000
    model = auto_model.from_pretrained(model_name).to(device)
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    loader = DataLoader(HFTokenDataset(token_ids, block_size), batch_size=batch_size, shuffle=True)
    if not len(loader):
        raise ValueError("corpus is too short for the selected block size")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(loader)
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = model(input_ids=batch, labels=batch).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # PyTorch serialization supports GPT-2's tied input/output embeddings on all
    # supported Transformers versions; older safetensors releases reject them.
    model.save_pretrained(output, safe_serialization=False)
    tokenizer.save_pretrained(output)
    stats = {
        "base_model": model_name,
        "device": str(device),
        "steps": max_steps,
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "duration_seconds": time.perf_counter() - started,
    }
    (output / "training_metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


@torch.no_grad()
def generate_pretrained(
    checkpoint: str | Path,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_k: int = 30,
    device_name: str = "auto",
) -> str:
    """Generate a coherent completion from a locally fine-tuned showcase checkpoint."""
    if temperature <= 0 or top_k < 1:
        raise ValueError("temperature and top_k must be positive")
    auto_model, auto_tokenizer = _transformers()
    device = select_device(device_name)
    tokenizer = auto_tokenizer.from_pretrained(checkpoint)
    model = auto_model.from_pretrained(checkpoint).to(device).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        min_length=input_ids.size(1) + min(20, max_new_tokens),
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
