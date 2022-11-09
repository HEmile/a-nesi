from typing import Tuple, List

import torch
from torch import nn
from torch.distributions import Categorical

from deepproblog.examples.MNIST.network import MNIST_Net
from gflownet import GFlowNetBase
from gflownet.experiments.state import MNISTAddState
from gflownet.gflownet import NeSyGFlowNet

EPS = 1E-6


class GFNMnistWMC(GFlowNetBase[MNISTAddState]):

    def __init__(self, N: int, hidden_size: int = 200, loss_f='bce-tb'):
        super().__init__(loss_f)
        self.n_classes = 2 * 10 ** N - 1
        self.N = N

        self.hidden_query = nn.Linear(20 * N, hidden_size)
        self.output_query = nn.Linear(hidden_size, self.n_classes)

        self.hiddens = nn.ModuleList([nn.Linear(20 * N + self.n_classes + i * 10, hidden_size) for i in range(2 * N)])
        self.outputs = nn.ModuleList([nn.Linear(hidden_size, 11) for _ in range(2 * N)])

    def flow(self, state: MNISTAddState) -> torch.Tensor:
        p = state.probability_vector().detach()
        if state.y is None:
            z = torch.relu(self.hidden_query(p))
            return torch.softmax(self.output_query(z), -1)

        # TODO: It has to recreate the one_hot vector every time, which is not efficient

        # TODO: This doesn't quite return a flow, but rather a distribution. Only in the case up here is it a flow.
        ds = state.state
        oh_query = state.oh_y
        inputs = torch.cat([p, oh_query] + state.oh_state, -1)
        z = torch.relu(self.hiddens[len(ds)](inputs))
        logits = self.outputs[len(ds)](z)
        # Predict amount of models for each digit
        return torch.softmax(logits[..., :-1], -1)


class MNISTAddModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.N = args["N"]
        # The NN that will model p(x) (digit classification probabilities)
        self.perception_network = MNIST_Net()
        hidden_size = args["hidden_size"]

        gfn = GFNMnistWMC(self.N, hidden_size, loss_f=args["loss"])
        # The Neurosymbolic GFlowNet that will perform the inference
        self.gfn: NeSyGFlowNet[MNISTAddState] = NeSyGFlowNet(gfn, prune=args["prune"], greedy_prob=args["greedy_prob"],
                                                             uniform_prob=args["uniform_prob"], loss_f=args["loss"])

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

        result = self.gfn.forward(initial_state, amt_samples=args['amt_samples'])

        # note: final_state is equal in both results
        state = result.final_state
        log_p = state.log_prob()

        success = state.success

        # Check if the sampled worlds are models. Only average over successful worlds (which is all in NeSy-GFN)
        total_successes = success.sum(-1)
        log_reward = torch.zeros_like(log_p)
        mask = total_successes > 0
        log_reward[mask] = (success[mask, :] * log_p[mask, :] / total_successes[mask].unsqueeze(-1))
        log_reward = log_reward.sum(-1)

        p_constraint = result.partitions[0]

        uncond_samples = Categorical(p).sample((100,))
        succes_p = (uncond_samples[..., 0] + uncond_samples[..., 1] == query).float().mean(0)

        # Use success probabilities as importance weights for the samples
        loss_p = -(p_constraint * log_reward).mean()
        loss_gfn = self.gfn.loss(result)
        return loss_p, loss_gfn, p_constraint, succes_p.mean()
