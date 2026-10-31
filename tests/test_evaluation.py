
import torch
from torch import nn

from src.evaluation import predict_classes


def test_predict_classes():
    model = nn.Identity()

    logits = torch.tensor([
        [2.0, 1.0],
        [0.2, 0.8],
        [0.9, 0.1],
    ])
    prediction = predict_classes(model, logits)
    print("pediction type", prediction)
    assert torch.equal(torch.tensor( [0, 1, 0]), prediction)



def test_predict_classes_sets_eval_mode():
    model = nn.Dropout(p=0.5)
    model.train()

    inputs = torch.tensor([
        [2.0, 1.0],
        [0.2, 0.8],
    ])

    predict_classes(model, inputs)
    model.eval()
    assert model.training == False



def test_predict_classes_disables_gradients():

    class GradCheckModel(nn.Module):
        def forward(self, x):
            self.grad_enabled_in_forward = torch.is_grad_enabled()
            return x

    model = GradCheckModel()

    inputs = torch.tensor([
        [2.0, 1.0],
        [0.2, 0.8],
    ], requires_grad=True)

    # TODO 1: Call predict_classes(model, inputs)
    predict_classes(model, inputs)
    # TODO 2: Assert that grad_enabled_in_forward is False
    assert model.grad_enabled_in_forward is  False



def test_predict_classes_device():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = nn.Linear(4, 3).to(device)
    inputs = torch.randn(2, 4).to(device)

    # TODO 1: Call predict_classes()
    prediction = predict_classes(model, inputs)
    # TODO 2: Check output shape == (2,)
    assert prediction.shape == (2,)
    # TODO 3: Check output device matches inputs.device
    assert prediction.device == inputs.device
