import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()    
        self.l1=nn.Linear(4,8)
        self.lu=nn.ReLU()
        self.l2=nn.Linear(8,2)


    def forward(self,x):
        x = self.l1(x)
        x = self.lu(x)
        x = self.l2(x)
        return x

x = torch.randn(8,4)
model = SimpleMLP()
output = model(x) 
print(output.shape)
