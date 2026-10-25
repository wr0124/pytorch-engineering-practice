import torch
from torch import nn

from src.checkpoints import load_model, save_model


def test_save_model(tmp_path):
    model = nn.Linear(4,2)
    pth = tmp_path/"model.pth"
    save_model(model, pth)
    assert pth.exists()


def test_load_model(tmp_path):
    model = torch.nn.Linear(4,2)
    pth = tmp_path/"model.pth" 
    save_model(model, pth)
    model2=torch.nn.Linear(4,2)
    model2_load = load_model(model2, pth)
    assert torch.equal(model2_load.weight, model.weight)
    assert torch.equal(model2_load.bias, model.bias)


def test_seed():
    torch.manual_seed(42)
    model = torch.nn.Linear(4,2)
    torch.manual_seed(42)
    model2 = torch.nn.Linear(4,2)
    assert torch.equal(model.weight, model2.weight)
    assert torch.equal(model.bias, model2.bias)
