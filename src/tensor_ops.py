import torch


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    x = torch.flatten(x, start_dim=1, end_dim=3)
    return x


def flatten_batch_reshape(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    return torch.reshape(x, (B, -1))


def channels_last(x: torch.Tensor) -> torch.Tensor:
    return torch.permute(x, (0, 2, 3, 1))


def spatial_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=(2, 3), keepdim=True)


def channel_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=1, keepdim=True)


def add_channel_dim(x: torch.Tensor) -> torch.Tensor:
    return torch.unsqueeze(x, 1)


def remove_channel_dim(x: torch.Tensor) -> torch.Tensor:
    return torch.squeeze(x, 1)


def first_image(x: torch.Tensor) -> torch.Tensor:
    return x[0:1, :, :, :]


def top_left_crop(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, 0:2, 0:2]


def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def normalize_images(x: torch.Tensor) -> torch.Tensor:
    x_mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
    x_std = torch.std(x, dim=(1, 2, 3), correction=0, keepdim=True)
    return (x - x_mean) / (x_std + 1e-8)
