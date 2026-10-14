import torch

from src.losses import compute_mse_loss


def test_compute_mse_loss():
    predictions = torch.tensor([2.0, 4.0, 6.0])
    targets = torch.tensor([1.0, 3.0, 5.0])
    loss = compute_mse_loss(predictions, targets)
    assert loss == 1.0 


def  test_compute_mse_loss_gradient():
    predictions = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    targets = torch.tensor([1.0, 3.0, 5.0])
    loss = compute_mse_loss(predictions, targets)
    loss.backward()
    torch.testing.assert_close( predictions.grad, torch.tensor([2/3, 2/3, 2/3])) 
