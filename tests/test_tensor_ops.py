import torch

from src.tensor_ops import channels_last, flatten_batch, flatten_batch_reshape


def test_flatten_batch():
    A = torch.ones(32,3,28,28)
    result = flatten_batch(A)
    assert result.shape == (32,2352)

def test_flatten_batch_reshape():
    A = torch.ones(8,3,64,64)
    result = flatten_batch_reshape(A)
    assert result.shape == (8, 12288)

def test_channels_last():
    A = torch.ones(8,3,64,32)
    result = channels_last(A)
    assert result.shape == (8, 64,32,3) 
