import torch
from torch.utils.data import DataLoader

from src.datasets import SimpleDataset

features = torch.randn(10, 3, 32, 32)
labels = torch.arange(10)

dataset = SimpleDataset(features, labels)

generator = torch.Generator()
generator.manual_seed(42)

loader = DataLoader( 	
		dataset,
		batch_size=4,
		shuffle=True,
	   	drop_last=False,
		generator=generator)


for epoch in range(2):
    print("Epoch:",  epoch)
    for images, labels in loader:
        print(labels)
