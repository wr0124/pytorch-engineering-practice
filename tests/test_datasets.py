import pytest
import torch

from src.datasets import SimpleDataset


def test_dataset_length():
    data = torch.randn(20, 3, 32, 32)
    label = torch.arange(0, 20)

    dataset = SimpleDataset(data, label)
    assert len(dataset) == 20


def test_dataset_getitem():
    data = torch.randn(10, 3, 32, 32)
    label = torch.arange(0, 10)
    dataset = SimpleDataset(data, label)
    image3, label3 = dataset[3]
    assert image3.shape == (3, 32, 32)
    assert label3 == 3
    assert torch.equal(image3, dataset.features[3])


def test_dataset_invalid_length():
    features = torch.randn(10, 3, 32, 32)
    label = torch.arange(0, 8)
    with pytest.raises(ValueError):
        SimpleDataset(features, label)
