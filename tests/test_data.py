import torch

from frankengpt.data import TokenDataset, load_corpora


def test_dataset_creates_shifted_next_token_pairs():
    dataset = TokenDataset([0, 1, 2, 3, 4], context_length=3)
    inputs, targets = dataset[0]
    assert len(dataset) == 2
    assert torch.equal(inputs, torch.tensor([0, 1, 2]))
    assert torch.equal(targets, torch.tensor([1, 2, 3]))


def test_load_corpora_combines_documents_with_boundary(tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    assert load_corpora([first, second]) == "first\n\n\n\nsecond"
    assert load_corpora([tmp_path / "*.txt"]) == "first\n\n\n\nsecond"
