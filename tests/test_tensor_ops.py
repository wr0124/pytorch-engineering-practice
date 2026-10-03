import torch

from src.tensor_ops import flatten_batch


def test_flatten_batch():
    A = torch.ones(32,3,28,28)
    result = flatten_batch(A)
    assert result.shape == (32,2352)
