import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_regression

x_numpy, y_numpy = make_regression(
    n_samples=50, n_features=1, noise=1, random_state=123
)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape
print(n_samples, n_features, y.shape)
input_dim = n_features
output_dim = 1
model = nn.Linear(input_dim, output_dim)

learning_rate = 1e-2
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000

for epoch in range(num_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(loss)

predicted = model(x).detach().numpy()
print(predicted.shape)

plt.plot(x_numpy, y_numpy, "ro")
plt.plot(x_numpy, predicted)
plt.savefig("tmp.png")
