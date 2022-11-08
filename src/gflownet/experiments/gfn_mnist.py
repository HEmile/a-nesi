import random
from typing import Tuple, List, Union, Literal

import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from deepproblog.examples.MNIST.network import MNIST_Net
from gflownet import GFlowNetBase
from gflownet.experiments import GFlowNetExact
from gflownet.experiments.state import MNISTAddState
from gflownet.gflownet import ST

EPS = 1E-6


class GFNMnist(GFlowNetBase[MNISTAddState]):

    def __init__(self, N: int, hidden_size: int = 200, memoizer_size=100, replay_size=5):
        super().__init__()
        if N > 1:
            raise NotImplementedError()
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

    def flow(self, state: MNISTAddState) -> torch.Tensor:
        if state.constraint is None:
            # TODO
            raise NotImplementedError()
        d1, _ = state.state
        query_oh = one_hot(state.constraint, self.n_classes).float()
        if len(d1) == 0:
            z1 = torch.relu(self.hidden_1(query_oh))
            # Predict amount of models for each digit
            return nn.functional.softplus(self.output_1(z1))
        else:
            d1_oh = one_hot(d1[0], 10).float()
            z2 = torch.relu(self.hidden_2(torch.cat([query_oh, d1_oh], -1)))
            return torch.sigmoid(self.output_2(z2))

    def loss(self, final_state: MNISTAddState, success: Tensor) -> Tensor:
        # Save new successes in memoizer
        b_i, s_i = success.nonzero(as_tuple=True)
        for b, s in zip(b_i, s_i):
            self.memoizer.append((final_state.query[b].item(), self.d1[s, b].item(), self.d2[s, b].item()))
            if len(self.memoizer) > self.memoizer_size:
                self.memoizer = self.memoizer[-self.memoizer_size:]

        # Naive monte carlo loss (with bootstrapping). No experience replay or whatever.
        # Non-sink RMSE loss
        f1_sel = self.f1.gather(-1, self.d1.T).T
        f2_sum = self.f2.sum(-1)
        l1 = ((f1_sel - f2_sum) ** 2).mean().sqrt()

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
            self.gfn = GFlowNetExact()
        else:
            self.gfn = GFNMnist(N, hidden_size)

    # Computes loss for a single batch
    def forward(self, d1: List[torch.Tensor], d2: List[torch.Tensor], query: torch.Tensor, amt_samples=1) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Generalize to N > 1
        N = 1
        # Predict the digit classification probabilities
        p1 = self.network(d1)
        p2 = self.network(d2)
        initial_state = MNISTAddState(([p1], [p2]), N, query)

        result = self.gfn.forward(initial_state, amt_samples)

        log_p = result.final_state.log_prob()

        success = result.final_state.success
        # Check if the sampled worlds are models. Only average over successful worlds
        log_reward = success * log_p / success.sum(-1).unsqueeze(-1)
        p_constraint = result.partitions[0]

        # Use success probabilities as importance weights for the samples
        loss_p = (-log_reward * p_constraint).mean()
        loss_gfn = self.gfn.loss(result)
        return loss_p, loss_gfn, p_constraint
