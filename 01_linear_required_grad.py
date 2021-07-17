import numpy as np
import torch
from sklearn.datasets import make_regression

x_numpy, y_numpy, coeff = make_regression(n_samples=100, n_features=1, noise=0, random_state=123, coef=True)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

print(x.shape, y.shape)

w = torch.tensor([0., 0.], dtype=torch.float32, requires_grad=True)

def forward(x):
    return w[0] + w[1] * x

def criterion(y_pred, y):
    return ((y_pred - y) ** 2).mean()

learning_rate = 0.01
num_epochs = 200

for epoch in range(num_epochs):
    y_pred = forward(x)
    loss = criterion(y_pred, y)  
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"weight: {w}; loss: {loss}")
    
print(f"true coeff: {coeff}")
