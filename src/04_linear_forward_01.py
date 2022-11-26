import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_regression

x_numpy, y_numpy, coeff = make_regression(
    n_samples=100, n_features=1, noise=0, random_state=123, coef=True
)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

print(x.shape, y.shape)

n_samples, n_features = x.shape
input_dim = n_features
output_dim = n_features
model = nn.Linear(input_dim, output_dim)
criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 200

for epoch in range(num_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        [weight, _] = model.parameters()
        print(f"weight: {weight[0][0]}; loss: {loss}")

print(f"true coeff: {coeff}")
