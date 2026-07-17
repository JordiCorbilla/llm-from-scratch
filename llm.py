"""Minimal GPT-style language model, training utilities, and text generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50_257
    context_length: int = 128
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.context_length < 2:
            raise ValueError("context_length must be >= 2")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    seed: int = 42


class TokenDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, token_ids: list[int], context_length: int, stride: int) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if len(token_ids) <= context_length:
            raise ValueError("corpus must contain more tokens than context_length")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.starts = list(range(0, len(token_ids) - context_length, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.starts[index]
        x = self.tokens[start : start + self.context_length]
        y = self.tokens[start + 1 : start + self.context_length + 1]
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.attention_dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, sequence, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        _, sequence = input_ids.shape
        if sequence > self.config.context_length:
            raise ValueError(f"sequence length {sequence} exceeds context length {self.config.context_length}")
        positions = torch.arange(sequence, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.no_grad()
def generate(
    model: GPTModel,
    input_ids: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.8,
    top_k: int | None = 40,
    eos_id: int | None = None,
) -> Tensor:
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        if temperature == 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -torch.inf
            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)
        if eos_id is not None and torch.all(next_id == eos_id):
            break
    return input_ids


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_learning_rate(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.no_grad()
def estimate_loss(model: GPTModel, loader: DataLoader, device: torch.device, batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for index, (x, y) in enumerate(loader):
        if index >= batches:
            break
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(
    path: Path,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: GPTConfig,
    train_config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
        path,
    )


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[GPTModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPTModel(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint
