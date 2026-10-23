import torch


def save_model(model, path):
    """
    Save a PyTorch model's parameters.

    Args:
        model: A PyTorch nn.Module.
        path: Destination file path.
    """

    torch.save(model.state_dict(), path)



