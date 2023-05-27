import torch

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets) -> None:
        super().__init__()
        self.features = features
        self.targets = targets


    def __getitem__(self, idx : int):
        return self.features[idx], self.targets[idx]
    
    def __len__(self) -> int:
        return len(self.targets)
