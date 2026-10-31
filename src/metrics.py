
import torch


def classification_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return classification accuracy as a scalar tensor."""

    
    preds = torch.argmax(logits, dim=1)
    
    diff = preds == targets
    
    mean_diff = diff.float().mean()
    return mean_diff

