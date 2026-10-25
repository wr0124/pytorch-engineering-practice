import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

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


def test_reprodicile_shuffling():
    data = torch.utils.data.TensorDataset(torch.arange(0,10))
    generator = torch.Generator()
    generator.manual_seed(42)

    loader = DataLoader(    
        data,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        generator=generator)
    
    batches = list(loader) 

    generator = torch.Generator()
    generator.manual_seed(42)
    
    loader2 = DataLoader(    
        data,
        batch_size=2, 
        shuffle=True,
        drop_last=False,
        generator=generator)
    batches2 = list(loader2)

    for i in batches:
        print(i, len(i))

    print( torch.cat([batch[0] for batch in batches])  )

    assert torch.equal( torch.cat([batch[0] for batch in batches]), torch.cat([batch[0] for batch in batches2]))



def test_missing_checkpoint(tmp_path ):
    path = tmp_path/"model.pth"
    model = torch.nn.Linear(4,2)
    with pytest.raises( FileNotFoundError):
        load_model(model, path)
