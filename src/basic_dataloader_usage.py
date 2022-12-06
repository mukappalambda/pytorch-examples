import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """
    Custom Dataset
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def main():
    """
    Example 1: Pass tensors directly into the DataLoader
    Example 2: Wrap the data with DataSet and pass the DataSet instance into the DataLoader
    """
    # Example 1
    n_samples = 50
    n_features = 4
    x = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples)
    dataloader1 = DataLoader(
        [[x[i], y[i]] for i in range(n_samples)],
        batch_size=12,
        shuffle=True,
    )

    for batch, (input, target) in enumerate(dataloader1):
        print(f"batch: {batch}")
        print(f"input size: {input.size()}")
        print(f"target size: {target.size()}")

    # Example 2
    dataset = DummyDataset(x=x, y=y)
    dataloader2 = DataLoader(
        dataset=dataset,
        batch_size=12,
        shuffle=True,
        drop_last=True,
    )

    for input, target in dataloader2:
        print(f"input shape: {input.shape}")
        print(f"target shape: {target.shape}")


if __name__ == "__main__":
    main()
