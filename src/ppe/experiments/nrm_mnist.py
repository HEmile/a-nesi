from typing import Tuple, List

import torch
from torch import nn
from torch.distributions import Categorical

from deepproblog.examples.MNIST.network import MNIST_Net
from ppe import NRMBase
from ppe.experiments.state import MNISTAddState

EPS = 1E-6


class NRMMnist(NRMBase[MNISTAddState]):

    def __init__(self, N: int, hidden_size: int = 200, loss_f='mse-tb'):
        super().__init__(loss_f)
        self.N = N

        hidden_queries = [nn.Linear(20 * N + (1 + (i - 1) * 10 if i >= 1 else 0), hidden_size) for i in range(N+1)]
        output_queries = [nn.Linear(hidden_size, 1)] + [nn.Linear(hidden_size, 10) for _ in range(N)]

        y_size = 1 + N * 10

        self.hiddens = nn.ModuleList(hidden_queries +
                                     [nn.Linear(20 * N + y_size + i * 10, hidden_size) for i in range(2 * N)])
        self.outputs = nn.ModuleList(output_queries +
                                     [nn.Linear(hidden_size, 10) for _ in range(2 * N)])

    def distribution(self, state: MNISTAddState) -> torch.Tensor:
        p = state.probability_vector().detach()
        layer_index = len(state.oh_state)
        inputs = torch.cat([p] + state.oh_state, -1)

        z = torch.relu(self.hiddens[layer_index](inputs))
        logits = self.outputs[layer_index](z)
        if len(state.oh_state) > 0:
            return torch.softmax(logits, -1)
        return torch.sigmoid(logits)


class MNISTAddModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.N = args["N"]
        # The NN that will model p(x) (digit classification probabilities)
        self.perception_network = MNIST_Net()
        hidden_size = args["hidden_size"]

        self.nrm = NRMMnist(self.N, hidden_size, loss_f=args["loss"])

    # Computes loss for a single batch
    def forward(self, MNISTd1: List[torch.Tensor], MNISTd2: List[torch.Tensor], query: torch.Tensor, args) -> \
    Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # TODO: Generalize to N > 1
        N = 1
        MNIST_in = torch.cat(MNISTd1 + MNISTd2, 1).reshape(-1, 1, 28, 28)
        # Predict the digit classification probabilities
        p = self.perception_network(MNIST_in).reshape(-1, 2 * N, 10)
        initial_state = MNISTAddState(p, N, query)

        result = self.nrm.forward(initial_state, amt_samples=args['amt_samples'])

        # note: final_state is equal in both results
        state = result.final_state
        log_p = state.log_p_world()

        success = state.success

        # Check if the sampled worlds are models. Only average over successful worlds (which is all in NeSy-GFN)
        total_successes = success.sum(-1)
        log_reward = torch.zeros_like(log_p)
        mask = total_successes > 0
        log_reward[mask] = (success[mask, :] * log_p[mask, :] / total_successes[mask].unsqueeze(-1))
        log_reward = log_reward.sum(-1)

        p_constraint = torch.stack(result.forward_probabilities[:N+1], -1).prod(-1)

        uncond_samples = Categorical(p).sample((100,))
        succes_p = (uncond_samples[..., 0] + uncond_samples[..., 1] == query).float().mean(0)

        # Use success probabilities as importance weights for the samples
        loss_p = -(p_constraint * log_reward).mean()
        loss_nrm = self.nrm.loss(result)
        return loss_p, loss_nrm, p_constraint, succes_p.mean()
