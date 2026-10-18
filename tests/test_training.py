import torch
from torch import nn

from src.training import train_step, validation_step


def test_train_step_updates_weights():
    model = nn.Linear(1,1,bias=False)

    with torch.no_grad():       
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    inputs = torch.tensor([[2.0]])
    targets = torch.tensor([[4.0]])
    weight_before = model.weight.detach().clone()
    train_step(model, optimizer, inputs,targets)  

    assert not torch.equal(model.weight, weight_before)
    torch.testing.assert_close(model.weight, torch.tensor( [[1.8]])   )
     

def test_validation_step_does_not_update_weights():
    model = nn.Linear(1,1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)     
    inputs = torch.tensor([[2.0]])
    targets = torch.tensor([[4.0]])
    weight_before = model.weight.detach().clone()
    loss = validation_step(model, inputs, targets) 
    assert torch.equal(model.weight, weight_before)
    torch.testing.assert_close(loss, torch.tensor(4.0))
    assert not loss.requires_grad
