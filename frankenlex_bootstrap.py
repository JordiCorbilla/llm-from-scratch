"""First-pass tokenizer + dataloader bootstrap for LLM-from-scratch practice."""

from pathlib import Path
import re
import sys
from urllib.request import urlopen

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
LOCAL_DATA_PATH = Path("data/pg84.txt")
TOKEN_PATTERN = r"([,.:;?_!\"()\[\]'`]|--|\s)"


def download_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def load_or_download_text(url: str, local_path: Path) -> str:
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    text = download_text(url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text, encoding="utf-8")
    return text


def normalize_text(text: str) -> str:
    return (
        text.replace("\ufeff", "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "--")
    )


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        text = normalize_text(text)
        preprocessed = re.split(TOKEN_PATTERN, text)
        preprocessed = [item.strip() for item in preprocessed if item and item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[token_id] for token_id in ids)
        text = re.sub(r"\s+([.:;?!\"()'])", r"\1", text)
        text = re.sub(r"'\s+(\w)", r"'\1", text)
        return text


class GPTDatasetV1(Dataset):
    """Creates sliding-window (input, target) token sequences for GPT training."""

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    suffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=suffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def print_coverage_stats(txt: str, max_length: int, stride: int) -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(txt)
    starts = list(range(0, len(token_ids) - max_length, stride))
    windows = len(starts)

    covered = [False] * max(len(token_ids) - 1, 0)
    for s in starts:
        end = min(s + max_length, len(covered))
        for j in range(s, end):
            covered[j] = True

    covered_positions = sum(covered)
    total_positions = len(token_ids) - 1
    coverage_ratio = (covered_positions / total_positions * 100) if total_positions > 0 else 0.0
    reuse_factor = (windows * max_length / covered_positions) if covered_positions > 0 else 0.0

    print(
        f"max_length={max_length}, stride={stride} | "
        f"windows={windows}, covered_positions={covered_positions}/{total_positions} "
        f"({coverage_ratio:.2f}%), reuse_factor={reuse_factor:.2f}x"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw_text = normalize_text(load_or_download_text(GUTENBERG_URL, LOCAL_DATA_PATH))
    print(f"Total number of characters: {len(raw_text)}")
    print(raw_text[:99])

    preprocessed = re.split(TOKEN_PATTERN, raw_text)
    preprocessed = [item.strip() for item in preprocessed if item and item.strip()]
    print(preprocessed[:30])

    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|unk|>", "<|endoftext|>"])

    vocab_size = len(all_tokens)
    print(vocab_size)
    print(all_tokens[:51])

    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    simple_tokenizer = SimpleTokenizerV2(vocab)

    text = "It's the last he did"
    ids = simple_tokenizer.encode(text)
    print(ids)
    print(simple_tokenizer.decode(ids))

    text_pair = "The modern prometheus . <|endoftext|> The Project Gutenberg ."
    pair_ids = simple_tokenizer.encode(text_pair)
    print(pair_ids)
    print(simple_tokenizer.decode(pair_ids))

    text_with_unknown = "Blorptastic quantum-capybara appears ."
    unk_ids = simple_tokenizer.encode(text_with_unknown)
    print(unk_ids)
    print(simple_tokenizer.decode(unk_ids))

    print(f"tiktoken version: {tiktoken.__version__}")
    bpe_tokenizer = tiktoken.get_encoding("gpt2")

    bpe_text = raw_text[:400] + " <|endoftext|> " + raw_text[400:480]
    bpe_ids = bpe_tokenizer.encode(bpe_text, allowed_special={"<|endoftext|>"})
    print(bpe_ids[:40])
    print(bpe_tokenizer.decode(bpe_ids) == bpe_text)

    unknown_sample = "Akwirw ier"
    unknown_ids = bpe_tokenizer.encode(unknown_sample)
    print(unknown_ids)
    for token_id in unknown_ids:
        print(f"{token_id} -> {bpe_tokenizer.decode([token_id])!r}")
    print(bpe_tokenizer.decode(unknown_ids))
    print(bpe_tokenizer.decode(unknown_ids) == unknown_sample)

    enc_text = bpe_tokenizer.encode(raw_text)
    print(f"Total number of tokens: {len(enc_text)}")

    enc_sample = enc_text[45:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    print()
    print("x:", x)
    print("y:", y)
    print()

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        target = enc_sample[i]
        print(f"context ids: {context} ---> target id: {target}")
        print(
            f"context text: {bpe_tokenizer.decode(context)!r} "
            f"---> next token text: {bpe_tokenizer.decode([target])!r}"
        )

    print("\n--- GPTDatasetV1 / DataLoader demo ---")
    dataloader = create_dataloader_v1(
        raw_text, batch_size=4, max_length=4, stride=1, suffle=False, drop_last=True, num_workers=0
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("first_batch:")
    print(first_batch)

    second_batch = next(data_iter)
    print("second_batch:")
    print(second_batch)

    print("Shift check (first_batch):")
    print(torch.equal(first_batch[0][:, 1:], first_batch[1][:, :-1]))

    print("\n--- Window behavior examples ---")
    print_coverage_stats(raw_text, max_length=2, stride=8)
    print_coverage_stats(raw_text, max_length=8, stride=2)
    print_coverage_stats(raw_text, max_length=8, stride=8)
    print_coverage_stats(raw_text, max_length=8, stride=1)


if __name__ == "__main__":
    main()
