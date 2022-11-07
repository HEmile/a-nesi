import random
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from deepproblog.examples.MNIST.network import MNIST_Net
from gflownet import GFlowNetBase
from gflownet.experiments import GFlowNetExact

EPS = 1E-6

class GFNMnist(GFlowNetBase):

    def __init__(self, N: int, hidden_size: int = 200, memoizer_size=100, replay_size=5):
        super().__init__()
        self.n_classes = 2 * 10 ** N - 1
        self.N = N
        # Assume N=1 for now
        self.hidden_1 = nn.Linear(self.n_classes, hidden_size)
        self.output_1 = nn.Linear(hidden_size, 10)

        self.hidden_2 = nn.Linear(self.n_classes + 10, hidden_size)
        self.output_2 = nn.Linear(hidden_size, 10)

        self.memoizer: List[Tuple[int, int, int]] = []
        self.replay_size = replay_size
        self.memoizer_size = memoizer_size

    def sample(self, p1: torch.Tensor, p2: torch.Tensor, query: torch.Tensor, amt_samples=1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Pass just the query
        query_oh = one_hot(query, self.n_classes).float()
        z1 = torch.relu(self.hidden_1(query_oh))
        # Predict amount of models for each digit
        self.f1 = nn.functional.softplus(self.output_1(z1))
        # Multiply with predicted digit probabilities
        unnorm_p1 = p1 * self.f1
        # Normalize and sample
        p1 = unnorm_p1 / unnorm_p1.sum(-1).unsqueeze(-1)
        self.d1 = Categorical(p1).sample((amt_samples,))

        d1_oh = one_hot(self.d1, 10).float()
        self.f2 = self._sink(d1_oh, query_oh.expand(amt_samples, -1, -1))
        unnorm_p2 = p2 * self.f2
        p2 = unnorm_p2 / unnorm_p2.sum(-1).unsqueeze(-1)
        self.d2 = Categorical(p2).sample()
        return [self.d1], [self.d2]

    def _sink(self, d1_oh: Tensor, q_oh: Tensor):
        z2 = torch.relu(self.hidden_2(torch.cat([q_oh, d1_oh], -1)))
        return torch.sigmoid(self.output_2(z2))

    def loss(self, success: Tensor, query: Tensor) -> Tensor:
        # Save new successes in memoizer
        b_i, s_i = success.nonzero(as_tuple=True)
        for b, s in zip(b_i, s_i):
            self.memoizer.append((query[b].item(), self.d1[s, b].item(), self.d2[s, b].item()))
            if len(self.memoizer) > self.memoizer_size:
                self.memoizer = self.memoizer[-self.memoizer_size:]

        # Naive monte carlo loss (with bootstrapping). No experience replay or whatever.
        # Non-sink RMSE loss
        f1_sel = self.f1.gather(-1, self.d1.T).T
        f2_sum = self.f2.sum(-1)
        l1 = ((f1_sel - f2_sum)**2).mean().sqrt()

        # Sink BCE loss
        f2_sel = self.f2.gather(-1, self.d2.unsqueeze(-1)).squeeze(-1).T
        l2 = nn.BCELoss()(f2_sel, success.float())

        if len(self.memoizer) == 0:
            return l1 + l2
        # Experience replay
        if len(self.memoizer) > self.replay_size:
            memory = random.choices(self.memoizer, k=self.replay_size)
        else:
            memory = self.memoizer
        # Compute loss on memory
        q_oh = one_hot(torch.tensor([m[0] for m in memory]), self.n_classes).float()
        d1_oh = one_hot(torch.tensor([m[1] for m in memory]), 10).float()
        f2_mem = self._sink(d1_oh, q_oh)
        d2_mem = torch.tensor([m[2] for m in memory])
        f2_sel = f2_mem.gather(-1, d2_mem.unsqueeze(-1)).squeeze(-1)
        l3 = nn.BCELoss()(f2_sel, torch.ones_like(f2_sel))
        return l1 + l2 + l3

class MNISTAddModel(nn.Module):
    gfn: GFlowNetBase

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
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # Predict the digit classification probabilities
        p1 = self.network(d1)
        p2 = self.network(d2)

        # Sample (hopefully) positive worlds to estimate gradients
        sample1_pos, sample2_pos = self.gfn.sample(p1, p2, query, amt_samples)

        # TODO: Generalize this to N > 1
        sample1_pos = sample1_pos[0].T
        sample2_pos = sample2_pos[0].T

        log_p_pos = ((p1 + EPS).log().gather(1, sample1_pos) + (p2 + EPS).log().gather(1, sample2_pos))

        success = None
        if self.method != "gfnexact":
            # Check if the sampled worlds are models
            success = sample1_pos + sample2_pos == query.expand(amt_samples, query.shape[0]).T
            log_p_pos *= success

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
        loss_p = (-log_p_pos * succes_p_smooth).mean()
        loss_gfn = self.gfn.loss(success, query)
        return loss_p, loss_gfn, succes_p
