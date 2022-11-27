import torch
from torch.distributions import Dirichlet
from torch.nn.functional import softplus


def fit_dirichlet(beliefs: torch.Tensor, alpha: torch.Tensor, lr=1, iters=1000) -> Dirichlet:
    a = softplus(alpha)
    data = beliefs
    N = data.shape[0]
    eps = 10e-8
    optimizer = torch.optim.Adam([alpha], lr=lr)
    statistics = (data + eps).log().mean(0).detach()
    for i in range(iters):
        # Dirichlet log likelihood. See https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
        # statistics = data.log().mean(0)
        # NOTE: The hardcoded version is quicker since the sufficient statistics are fixed.
        # NOTE: We have to think about how to parallelize this. Should be trivial (especially if the dimensions of each dirichlet is fixed)
        log_p = torch.lgamma(a.sum(-1) + eps) - \
                torch.lgamma(a + eps).sum(-1) + \
                torch.sum((a - 1) * statistics, -1)
        log_p = log_p * N
        optimizer.zero_grad()

        loss = -log_p.mean()

        loss.backward(retain_graph=True)
        optimizer.step()
        # Sometimes the algorithm will find negative numbers during minimizing the log probability.
        # However alpha needs to be positive.
        a = softplus(alpha)

    return Dirichlet(a)