import torch
from torch.distributions import Dirichlet
from torch.nn.functional import softplus


def fit_dirichlet(beliefs: torch.Tensor, alpha: torch.Tensor, optimizer, iters=1000, L2=0.0) -> Dirichlet:
    """
    Fit a Dirichlet distribution to the beliefs.
    :param beliefs: Tensor of shape (K, |W|, n) where K is the number of beliefs, |W| is number of elements in a world
     and n is the number of classes.
    :param alpha: Tensor of shape (|W|, n) of the prior.
    :param lr: Learning rate for alpha
    :param iters: Number of iterations to optimize log-probability of Dirichlet
    :param L2: L2 regularization on alpha. If 0, no regularization. If > 0, this will prefer lower values of alpha.
     This is used to prevent the Dirichlet distribution from becoming too peaked on the uniform distribution over classes
    """
    a = softplus(alpha)
    data = beliefs
    N = data.shape[0]
    eps = 10e-8
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

        loss = -log_p.mean() + L2 * (a ** 2).mean()

        loss.backward(retain_graph=True)
        optimizer.step()
        # Sometimes the algorithm will find negative numbers during minimizing the log probability.
        # However alpha needs to be positive.
        a = softplus(alpha)

    return Dirichlet(a)