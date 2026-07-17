import torch

from frankengpt.data import TokenDataset


def test_dataset_creates_shifted_next_token_pairs():
    dataset = TokenDataset([0, 1, 2, 3, 4], context_length=3)
    inputs, targets = dataset[0]
    assert len(dataset) == 2
    assert torch.equal(inputs, torch.tensor([0, 1, 2]))
    assert torch.equal(targets, torch.tensor([1, 2, 3]))
