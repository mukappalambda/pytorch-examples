import torch
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = x.shape
input_dim = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_dim=input_dim)

criterion = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 200
for epoch in range(num_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if epoch % 2 == 0:
    #     print(f"epoch: {epoch}; weight: {model.state_dict()}; loss: {loss}")

model_path = "lin_reg_model_weights.pth"
torch.save(model.state_dict(), model_path)

second_model = LinearRegression(input_dim=input_dim)
second_model.load_state_dict(torch.load(model_path))
second_model.eval()

y_pred = second_model(x).detach()
resid = (y_pred - y).mean()
print(f"y_pred: {y_pred}; residual: {resid}")

