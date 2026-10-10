from torch.utils.data import Dataset

class SimpleDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("Features and labels must have the same number of samples")

    def __len__(self):
        return len(self.features)

    def __getitem__(self,index):
        return self.features[index], self.labels[index]

    

