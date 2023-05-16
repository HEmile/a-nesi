import torch


def log1mexp(x):
    assert (torch.all(x >= 0))
    EPS = 1e-15
    out = torch.ones_like(x)
    cond1 = x <= 0.6931471805599453094
    out[cond1] = torch.log(-torch.expm1(-x[cond1]) + EPS)
    out[~cond1] = torch.log1p(-torch.exp(-x[~cond1]) + EPS)
    return out

def log_not(log_p: torch.Tensor):
    return log1mexp(-log_p)