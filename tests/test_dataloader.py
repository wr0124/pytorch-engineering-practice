import torch
from torch.utils.data import DataLoader

from src.datasets import SimpleDataset


def test_dataloader_batch_count():
    features = torch.randn(23,3,32,32)
    labels = torch.arange(0,23)
    dataset = SimpleDataset(features,labels)
    loader = DataLoader(dataset, batch_size=5,shuffle=False,drop_last=False)
    assert len(loader) == 5


def test_dataloader_last_batch_size():
    features = torch.randn(23,3,32,32)
    labels = torch.arange(0,23)
    dataset = SimpleDataset(features,labels)
    loader = DataLoader(dataset, batch_size=5,shuffle=False,drop_last=False)
    batches = list(loader)
    image, _ = batches[-1] 
    assert image.shape[0] == 3


def test_dataloader_drop_last():
    features = torch.randn(23,3,32,32)
    labels = torch.arange(0,23)
    dataset = SimpleDataset(features,labels)
    loader = DataLoader(dataset, batch_size=5,shuffle=False,drop_last=True)
    assert len(loader) == 4
    last_batch = list(loader)[-1]
    images, labels = last_batch
    assert images.shape[0] == 5


