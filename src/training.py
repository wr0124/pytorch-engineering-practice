import torch
from torch import nn

from src.losses import compute_mse_loss


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = compute_mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()
    return loss
