import torch


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    x = torch.flatten(x, start_dim=1,end_dim=3)
    return x

def flatten_batch_reshape(x:torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    return torch.reshape(x, (B, -1))
