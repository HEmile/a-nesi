from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from deepproblog.examples.MNIST.network import MNIST_Net
from gflownet import GFlowNetBase
from gflownet.experiments import GFlowNetExact


class GFNMnist(GFlowNetBase):

    def __init__(self, N: int, hidden_size: int = 200):
        super().__init__()
        self.n_classes = 2 * 10 ** N - 1
        self.N = N
        # Assume N=1 for now
        self.hidden_1 = nn.Linear(self.n_classes, hidden_size)
        self.output_1 = nn.Linear(hidden_size, 10)

        self.hidden_2 = nn.Linear(self.n_classes + 10, hidden_size)
        self.output_2 = nn.Linear(hidden_size, 10)

    def sample(self, p1: torch.Tensor, p2: torch.Tensor, query: torch.Tensor, amt_samples=1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Pass just the query
        query_oh = torch.nn.functional.one_hot(query, self.n_classes).float()
        z1 = torch.relu(self.hidden_1(query_oh))
        # Predict amount of models for each digit
        f1 = nn.functional.softplus(self.output_1(z1))
        # Multiply with predicted digit probabilities
        unnorm_p1 = p1 * f1
        # Normalize and sample
        p1 = unnorm_p1 / unnorm_p1.sum(-1).unsqueeze(-1)
        d1 = Categorical(p1).sample((amt_samples,))

        d1_oh = torch.nn.functional.one_hot(d1, 10).float()
        z2 = torch.relu(self.hidden_2(torch.cat([query_oh.expand(amt_samples, -1, -1), d1_oh], -1)))
        f2 = torch.sigmoid(self.output_2(z2))
        unnorm_p2 = p2 * f2
        p2 = unnorm_p2 / unnorm_p2.sum(-1).unsqueeze(-1)
        d2 = Categorical(p2).sample()
        return [d1], [d2]


class MNISTAddModel(nn.Module):
    def __init__(self, N: int, method: str, hidden_size: int = 200):
        super().__init__()
        self.N = N
        # The NN that will model p(x) (digit classification probabilities)
        self.network = MNIST_Net()
        self.method = method
        if method == 'gfnexact':
            self.gfn = GFlowNetExact(N)
        else:
            self.gfn = GFNMnist(N, hidden_size)

    # Computes loss for a single batch
    def forward(self, d1: List[torch.Tensor], d2: List[torch.Tensor], query: torch.Tensor, amt_samples=1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # Predict the digit classification probabilities
        p1 = self.network(d1)
        p2 = self.network(d2)

        # Sample (hopefully) positive worlds to estimate gradients
        sample1_pos, sample2_pos = self.gfn.sample(p1, p2, query, amt_samples)

        # TODO: Generalize this to N > 1
        sample1_pos = sample1_pos[0].T
        sample2_pos = sample2_pos[0].T

        log_p_pos = (p1.log().gather(1, sample1_pos) + p2.log().gather(1, sample2_pos))

        if self.method != "gfnexact":
            # Check if the sampled worlds are models
            log_p_pos *= sample1_pos + sample2_pos == query.expand(amt_samples, query.shape[0]).T

        log_p_pos = log_p_pos.mean(1)
        # Smoothed mc succes probability estimate. Smoothed to ensure positive samples aren't ignored, but obv biased
        # Should maybe test if unbiased estimation works as well
        sample_r1 = Categorical(p1).sample((2 * amt_samples,))
        sample_r2 = Categorical(p2).sample((2 * amt_samples,))

        # Note: This isn't cheating, it just evaluates 'the program' in parallel
        corr_counts = (sample_r1 + sample_r2 == query).float().sum(0)
        succes_p = corr_counts / (2 * amt_samples)
        succes_p_smooth = (corr_counts + 1) / (2 * amt_samples + 2)

        # Use success probabilities as importance weights for the samples
        loss = (-log_p_pos * succes_p_smooth).mean()
        return loss, succes_p
