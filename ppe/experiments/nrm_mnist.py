from typing import Optional

import torch
from torch import nn

from nrm import NRMResult
from experiments.MNISTNet import MNIST_Net
from nrm import NRMBase
from experiments.state import MNISTAddState
from ppe import PPEBase

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
        p = state.probability_vector()  # .detach()
        layer_index = len(state.oh_state)
        inputs = torch.cat([p] + state.oh_state, -1)

        z = torch.relu(self.hiddens[layer_index](inputs))
        logits = self.outputs[layer_index](z)
        if len(state.oh_state) > 0:
            return torch.softmax(logits, -1)
        return torch.sigmoid(logits)


class MNISTAddModel(PPEBase[MNISTAddState]):

    def __init__(self, args):
        self.N = args["N"]
        # The NN that will model p(x) (digit classification probabilities)
        hidden_size = args["hidden_size"]

        nrm = NRMMnist(self.N, hidden_size, loss_f=args["loss"])
        super().__init__(nrm, MNIST_Net(),
                         amount_samples=args['amt_samples'],
                         belief_size=[10] * 2 * self.N,
                         dirichlet_lr=args['dirichlet_lr'],
                         dirichlet_iters=args['dirichlet_iters'],
                         initial_concentration=args['dirichlet_init'],
                         K_beliefs=args['K_beliefs'],
                         nrm_lr=args['nrm_lr'],
                         nrm_loss=args['nrm_loss'],
                         perception_lr=args['perception_lr'],
                         perception_loss=args['perception_loss'],)

    def initial_state(self, P: torch.Tensor, y: Optional[torch.Tensor] = None, w: Optional[torch.Tensor] = None,
                      generate_w=True) -> MNISTAddState:
        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(self.N * 2)]
        y_list = None
        if y is not None:
            y_list = [torch.floor(y / (10 ** (self.N - i)) % 10).long() for i in range(self.N + 1)]
        return MNISTAddState(P, self.N, (y_list, w_list), generate_w=generate_w)

    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (batch_size, 2*n)
        """
        stack1 = torch.stack([10 ** (self.N - i - 1) * w[:, i] for i in range(self.N)], -1)
        stack2 = torch.stack([10 ** (self.N - i - 1) * w[:, self.N + i] for i in range(self.N)], -1)

        n1 = stack1.sum(-1)
        n2 = stack2.sum(-1)

        return n1 + n2

    def success(self, result: NRMResult[MNISTAddState], y: torch.Tensor) -> torch.Tensor:
        sample_y = result.final_state.y
        stack = torch.stack([10 ** (self.N - i) * sample_y[i] for i in range(self.N + 1)], -1)
        return stack.sum(-1) == y.unsqueeze(-1)
