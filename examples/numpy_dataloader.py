import numpy as np
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def main():
    n_samples = 50
    n_features = 4
    x = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    dataloader0 = DataLoader(
        dataset=[(a, b) for a, b in zip(x, y)],
        batch_size=10,
        shuffle=True,
    )

    for batch, (input_, target_) in enumerate(dataloader0):
        print(f"batch: {batch}")
        print(input_)
        print(target_)

    dataset = MyDataset(x=x, y=y)
    dataloader1 = DataLoader(
        dataset=dataset,
        batch_size=12,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    for batch, (input_, target_) in enumerate(dataloader1):
        print(f"batch: {batch}")
        print(input_)
        print(target_)


if __name__ == "__main__":
    main()
