import torch

from src.models import SimpleMLP


def test_simple_mlp_output_shape():
    mlp_model = SimpleMLP()
    inputs = torch.randn(8,4)
    outputs = mlp_model(inputs)
    assert outputs.shape == (8,2)


def test_simple_mlp_parameter_count():
     mlp_model = SimpleMLP()
     total = sum( number.numel() for number in  mlp_model.parameters()  ) 
     assert total == 58


def test_simple_mlp_gradients():
    mlp_model = SimpleMLP()
    inputs = torch.randn(8,4)
    outputs = mlp_model(inputs)
    loss = outputs.sum()
    loss.backward()
    for param in mlp_model.parameters():
        assert param.grad is not None


def test_eval_mode_allows_gradients():
    mlp_model = SimpleMLP()
    mlp_model.eval()
    inputs = torch.randn(8,4)
    outputs = mlp_model(inputs)
    assert outputs.requires_grad


def test_no_grad_disables_gradient_tracking():
    mlp_model = SimpleMLP()
    mlp_model.eval()
    inputs = torch.randn(8,4)
    with torch.no_grad():
        outputs = mlp_model(inputs)
    assert not outputs.requires_grad
