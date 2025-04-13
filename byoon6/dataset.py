from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
