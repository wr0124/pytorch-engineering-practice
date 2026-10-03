import torch


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    x = torch.flatten(x, start_dim=1,end_dim=3)
    return x


