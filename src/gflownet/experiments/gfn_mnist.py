from typing import Tuple, List

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from deepproblog.examples.MNIST.network import MNIST_Net
from gflownet import GFlowNetBase
from gflownet.experiments import GFlowNetExact
from gflownet.experiments.state import MNISTAddState
from gflownet.gflownet import NeSyGFlowNet

EPS = 1E-6


class GFNMnistMC(GFlowNetBase[MNISTAddState]):

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


class GFNMnistWMC(GFlowNetBase[MNISTAddState]):

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


class MNISTAddModel(nn.Module):

    def __init__(self, N: int, mc_method: str, hidden_size: int = 200):
        super().__init__()
        self.N = N
        # The NN that will model p(x) (digit classification probabilities)
        self.perception_network = MNIST_Net()
        if mc_method == 'gfnexact':
            gfn_mc = GFlowNetExact()
        else:
            gfn_mc = GFNMnistMC(N, hidden_size)
        gfn_wmc = GFNMnistWMC(N, hidden_size)
        self.gfn: NeSyGFlowNet[MNISTAddState] = NeSyGFlowNet(gfn_mc, gfn_wmc)

    # Computes loss for a single batch
    def forward(self, MNISTd1: List[torch.Tensor], MNISTd2: List[torch.Tensor], query: torch.Tensor, amt_samples=1) -> \
    Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Generalize to N > 1
        N = 1
        # Predict the digit classification probabilities
        p1 = self.perception_network(MNISTd1)
        p2 = self.perception_network(MNISTd2)
        initial_state = MNISTAddState(([p1], [p2]), N, query)

        result_mc, result_wmc = self.gfn.forward(initial_state, amt_samples=amt_samples, sampler=self.greedy_sampler)

        # note: final_state is equal in both results
        state = result_mc.final_state
        log_p = state.log_prob()

        success = state.success
        # Check if the sampled worlds are models. Only average over successful worlds
        log_reward = (success * log_p / success.sum(-1).unsqueeze(-1)).sum(-1)

        p_constraint = result_wmc.partitions[0]

        # Use success probabilities as importance weights for the samples
        loss_p = (-log_reward * p_constraint).mean()
        loss_mc_gfn, loss_wmc_gfn = self.gfn.loss(result_mc, result_wmc)
        return loss_p, loss_mc_gfn, loss_wmc_gfn, p_constraint
