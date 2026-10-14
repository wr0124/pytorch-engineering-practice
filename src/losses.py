import torch


def compute_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    mse_loss = torch.mean((predictions - targets)**2)
    return mse_loss
