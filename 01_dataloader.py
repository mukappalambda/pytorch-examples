import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_regression

x_numpy, y_numpy = make_regression(n_samples=500, n_features=1, noise=1, random_state=123)

class DummyDataset(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(x_numpy)
        self.y = torch.from_numpy(y_numpy)
        self.n_samples = x_numpy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

batch_size = 50
dataset = DummyDataset()
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)

first_batch = torch.from_numpy(x_numpy[:batch_size, :])
second_batch = torch.from_numpy(x_numpy[batch_size:2 * batch_size, :])

dataiter = iter(dataloader)
data = dataiter.next()
samples, labels = data
# print(f"samples: {samples}; labels: {labels}")
print(f"first batch shape: {first_batch.shape}; first samples shape: {samples.shape}")
assert torch.equal(first_batch, samples)
# print(f"samples shape: {samples.shape}; labels shape: {labels.shape}")

data = dataiter.next()
samples, labels = data
assert torch.equal(second_batch, samples)

# num_epochs = 2
# dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
# num_iters = len(dataset) // batch_size

# for epoch in range(num_epochs):
#     for it, (samples, labels) in enumerate(dataloader):
#         print(f"epoch: {epoch + 1}; iteration: {it}; samples shape: {samples.shape}; labels shape: {labels.shape}")
