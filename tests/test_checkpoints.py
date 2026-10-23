from torch import nn

from src.checkpoints import save_model


def test_save_model(tmp_path):
    model = nn.Linear(4,2)
    pth = tmp_path/"model.pth"
    save_model(model, pth)
    assert pth.exists()
