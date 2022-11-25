import torch

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor([0], dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


def loss(y, y_pred):
    return ((y - y_pred) ** 2).mean()


def gradient(x, y, y_pred):
    return torch.dot(2 * x, y_pred - y).mean()


learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    y_pred = forward(x)
    l = loss(y, y_pred)
    # dw = gradient(x, y, y_pred)
    l.backward()

    # w -= learning_rate * dw
    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    if epoch % 2 == 0:
        print(f"weight: {w}; loss: {l}")


# design model
# forward pass
# backward pass
# step
