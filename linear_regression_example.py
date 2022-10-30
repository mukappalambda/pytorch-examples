import numpy as np
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 10, 100)
beta0 = 0.2
beta1 = 0.3
y = beta0 + beta1 * x

x = x[:, np.newaxis]

# method 1
lr = LinearRegression()
lr.fit(x, y)
print(lr.intercept_, lr.coef_)

# method 2
ones = np.ones_like(x)
A = np.hstack([ones, x])
betahat = np.linalg.pinv(A) @ y
print(betahat)

# method 3
def grad_descent(beta, x, y):
    x = x.reshape(
        -1,
    )
    n = len(y)
    partial_0 = (-2 / n) * np.sum(y - beta[0] - beta[1] * x)
    partial_1 = (-2 / n) * np.sum(x * (y - beta[0] - beta[1] * x))
    return np.array([partial_0, partial_1])


beta_next = np.random.randn(2)

for _ in range(1000):
    grad = grad_descent(beta_next, x, y)
    beta_next -= 1e-2 * grad

print(beta_next)
