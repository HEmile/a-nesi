import torch
from torch.distributions import Dirichlet

# data = torch.tensor([[0.1,  0.5,  0.3,  0.05, 0.05],
#                      [0.8,  0.01, 0.01, 0.08, 0.1],
#                      [0.1,  0.1,  0.1,  0.1,  0.6],
#                      [0.01, 0.94, 0.01, 0.01, 0.03]])
# initial_a = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], requires_grad=True)
#
# N = data.shape[0]
# a = initial_a
# print(data.sum(1))
# optimizer = torch.optim.Adam([a], lr=0.1)
#
# def log_likelihood(data, a):
#     d = Dirichlet(a)
#     return d.log_prob(data).sum()
#
# statistics = data.log().mean(0)
# for i in range(100):
#     # Dirichlet log likelihood. See https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
#     # statistics = data.log().mean(0)
#     # NOTE: The hardcoded version is quicker since the sufficient statistics are fixed.
#     # NOTE: We have to think about how to parallelize this. Should be trivial (especially if the dimensions of each dirichlet is fixed)
#     log_p = N * torch.lgamma(a.sum()) - \
#             N * torch.lgamma(a).sum() + \
#             N * torch.sum((a - 1) * statistics)
#     optimizer.zero_grad()
#
#     loss = -log_p
#
#     if i %10 == 0:
#         print(f"Loss: {loss.item()}")
#         print(a)
#     loss.backward()
#     optimizer.step()
#
# for i in range(10):
#     print(Dirichlet(a).sample())


def fit_dirichlet(beliefs: torch.Tensor) -> Dirichlet:
    pass