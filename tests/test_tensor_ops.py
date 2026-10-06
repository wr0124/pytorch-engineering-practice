import torch

from src.tensor_ops import (
    channels_last,
    flatten_batch,
    flatten_batch_reshape,
    spatial_mean,
)


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

def test_spatial_mean():
    A = torch.Tensor(
	 [
	 	[ [[1, 2, 3], [1, 2, 3]] ,  [[4, 5, 6], [4, 5, 6]] ] ,  
	  
		[ [[1, 2, 3], [1, 2, 3]] , [[4, 5, 6], [4, 5, 6]] ]  ,
        ] 
	)
    result = spatial_mean(A)
    assert  torch.equal ( result , torch.Tensor( [ [[ [2]] ,  [ [5]]  ], [[ [2]] ,  [ [5]]  ]  ] ) )
