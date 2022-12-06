import torch
from torch import distributions as D
from torch import optim
from torch.utils.data import DataLoader


def forward(
    data: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the negative log likelihood.
    """
    dist_ = D.Normal(mu, torch.abs(sigma))
    nll = -dist_.log_prob(data).sum()
    return nll


def main():
    """
    An example of ML estimation of the 1D Gaussian distribution.
    """
    mu_true_ = torch.tensor(123.0, dtype=torch.float64, requires_grad=True)
    sigma_true_ = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    n_samples = 5000

    m = D.Normal(mu_true_, sigma_true_)
    samples = m.sample((n_samples,))
    dataloader = DataLoader(
        samples,
        batch_size=64,
        shuffle=True,
    )

    mu_ = torch.randn(1, requires_grad=True)
    signed_sigma_ = torch.randn(1, requires_grad=True)
    optimizer = optim.RMSprop([mu_, signed_sigma_], lr=1e-3, momentum=0.9)

    n_epochs = 200
    for epoch in range(n_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            nll = forward(data, mu_, signed_sigma_)
            nll.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}")
            print(f"mu: {mu_.item()}; sigma: {torch.abs(signed_sigma_).item()}")


if __name__ == "__main__":
    main()
