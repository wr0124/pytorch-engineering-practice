import torch

from src.metrics import classification_accuracy


def test_classification_accuracy():
    
    logits = torch.tensor([
    [2.0, 1.0],
    [0.2, 0.8],
    [0.9, 0.1],
    ])

    targets = torch.tensor([0, 0, 0])
    
    accuracy = classification_accuracy(logits, targets)
    
    assert torch.isclose(accuracy, torch.tensor(2/3))
    assert accuracy.shape == torch.Size([])



def test_classification_accuracy_perfect():
    logits = torch.tensor([
    [3.0, 1.0],
    [0.2, 2.0],
    [4.0, 0.1],
])

    targets = torch.tensor([0, 1, 0])
    accuracy = classification_accuracy(logits, targets)
    assert accuracy == 1.0


def test_classification_accuracy_device():
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu" )
    logits = torch.randn(2,2).to(device)
    targets = torch.tensor([0, 1]).to(device)
    accuracy = classification_accuracy(logits, targets)
    assert accuracy.device == logits.device
    
