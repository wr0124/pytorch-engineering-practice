
import torch
from torch import nn


def predict_classes(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Return predicted class indices without tracking gradients."""

    model.eval()
    with torch.no_grad():
        output=model(inputs)
    indices = torch.argmax(output, dim=1)
    return indices
