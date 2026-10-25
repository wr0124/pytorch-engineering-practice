import torch


def save_model(model, path):
    """
    Save a PyTorch model's parameters.

    Args:
        model: A PyTorch nn.Module.
        path: Destination file path.
    """

    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load saved parameters into an existing model.

    Args:
        model: An existing PyTorch nn.Module.
        path: Path to a saved checkpoint.
    """

    state = torch.load(path, weights_only = True)
    model.load_state_dict(state)
    return model

