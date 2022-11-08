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
        self.n_classes = 2 * 10 ** N - 1
        self.N = N

        self.query_mc = torch.nn.Parameter(torch.randn(1, self.n_classes) + 2, requires_grad=True)

        self.hiddens = nn.ModuleList([nn.Linear(self.n_classes + i * 10, hidden_size) for i in range(2 * N)])
        self.outputs = nn.ModuleList([nn.Linear(hidden_size, 10) for _ in range(2 * N)])

        # TODO: Reimplement this (properly, with only saving unique memos)
        self.memoizer: List[Tuple[int, int, int]] = []
        self.replay_size = replay_size
        self.memoizer_size = memoizer_size

    def flow(self, state: MNISTAddState) -> torch.Tensor:
        if state.constraint is None:
            return nn.functional.softplus(self.query_mc)

        ds = state.state
        inputs = torch.cat([state.oh_query] + state.oh_state, -1)
        z = torch.relu(self.hiddens[len(ds)](inputs))
        # Predict amount of models for each digit
        return nn.functional.softplus(self.outputs[len(ds)](z))


class GFNMnistWMC(GFlowNetBase[MNISTAddState]):

    def __init__(self, N: int, hidden_size: int = 200):
        super().__init__()
        self.n_classes = 2 * 10 ** N - 1
        self.N = N

        self.hidden_query = nn.Linear(20 * N, hidden_size)
        self.output_query = nn.Linear(hidden_size, self.n_classes)

        self.hiddens = nn.ModuleList([nn.Linear(20 * N + self.n_classes + i * 10, hidden_size) for i in range(2 * N)])
        self.outputs = nn.ModuleList([nn.Linear(hidden_size, 10) for _ in range(2 * N)])

    def flow(self, state: MNISTAddState) -> torch.Tensor:
        p = state.probability_vector()
        if state.constraint is None:
            z = torch.relu(self.hidden_query(p))
            return torch.sigmoid(self.output_query(z))

        # TODO: It has to recreate the one_hot vector every time, which is not efficient

        ds = state.state
        oh_query = state.oh_query
        inputs = torch.cat([p, oh_query] + state.oh_state, -1)
        z = torch.relu(self.hiddens[len(ds)](inputs))
        # Predict amount of models for each digit
        return nn.functional.softplus(self.outputs[len(ds)](z))


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
        MNIST_in = torch.cat(MNISTd1 + MNISTd2, 1).reshape(-1, 1, 28, 28)
        # Predict the digit classification probabilities
        p = self.perception_network(MNIST_in).reshape(-1, 2* N, 10)
        initial_state = MNISTAddState(p, N, query)

        result_mc, result_wmc = self.gfn.forward(initial_state, amt_samples=amt_samples)

        # note: final_state is equal in both results
        state = result_mc.final_state
        log_p = state.log_prob()

        success = state.success
        # Check if the sampled worlds are models. Only average over successful worlds
        log_reward = (success * log_p / success.sum(-1).unsqueeze(-1)).sum(-1)

        p_constraint = result_wmc.partitions[0]

        # Use success probabilities as importance weights for the samples
        loss_p = (-log_reward * p_constraint).mean()
        loss_mc_gfn, loss_wmc_gfn = self.gfn.joint_loss(result_mc, result_wmc)
        return loss_p, loss_mc_gfn, loss_wmc_gfn, p_constraint
