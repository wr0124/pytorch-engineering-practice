import torch

from src.tensor_ops import (
    add_channel_dim,
    channel_mean,
    channels_last,
    first_image,
    flatten_batch,
    flatten_batch_reshape,
    remove_channel_dim,
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

def test_channel_mean():
    A = torch.Tensor(
        [
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ]
        ]
    )

    result = channel_mean(A)

    expected = torch.Tensor(
        [
            [
                [[3, 4], [5, 6]]
            ]
        ]
    )

    assert torch.equal(result, expected)

def test_add_channel_dim():
    A=torch.Tensor(
	[
	    [
		[1,2,3],
		[4,5,6]
	    ],
	    [
		[7,8,9],
		[10,11,12]
	    ]
	]
    )
    result = add_channel_dim(A)
    expected = torch.Tensor( 
	[
	    [  [ [1,2,3],[4,5,6] ]    ],

	    [  [ [7,8,9],[10,11,12] ]    ]

	]
    )
    assert torch.equal(result, expected)


def test_remove_channel_dim():
    A=torch.Tensor(
	[ 
	    [
	    [
		[1,2,3],
		[4,5,6]
	    ]
	    ]
	]
    )
    result = remove_channel_dim(A)
    expected = torch.Tensor(
	[
	    [
		[1,2,3],
		[4,5,6]
	    ]

	]
	)
    assert torch.equal(result, expected)


def test_first_image( ):
    A=torch.Tensor(
	[
	    [
		[
		[1,2],[3,4]	
		]
	    ],
            [
                [
                [5,6],[7,8]
                ]
            ]
	]
	)
    result = first_image(A)
    expected = torch.Tensor(
        [   
            [   
                [
                [1,2],[3,4]
                ]
            ]
	]
	)
    assert torch.equal(result, expected)
