from typing import Optional, List

import torch
from torch import nn

from nrm import NRMResult
from experiments.MNISTNet import MNIST_Net
from nrm import NRMBase
from experiments.state import MNISTAddState
from ppe import PPEBase

EPS = 1E-6


class NRMMnist(NRMBase[MNISTAddState]):

    def __init__(self, N: int, layers = 1, hidden_size: int = 200, prune: bool = True):
        super().__init__(prune)
        self.N = N
        self.layers = layers

        hidden_queries = [nn.Linear(20 * N + (1 + (i - 1) * 10 if i >= 1 else 0), hidden_size) for i in range(N + 1)]
        output_queries = [nn.Linear(hidden_size, 1)] + [nn.Linear(hidden_size, 10) for _ in range(N)]

        y_size = 1 + N * 10

        self.input_layer = nn.ModuleList(hidden_queries +
                                     [nn.Linear(20 * N + y_size + i * 10, hidden_size) for i in range(2 * N)])
        self.hiddens = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range((3 * N + 1) * (layers - 1))])
        self.outputs = nn.ModuleList(output_queries +
                                     [nn.Linear(hidden_size, 10) for _ in range(2 * N)])

    def distribution(self, state: MNISTAddState) -> torch.Tensor:
        p = state.probability_vector()  # .detach()
        layer_index = len(state.oh_state)
        inputs = torch.cat([p] + state.oh_state, -1)

        z = torch.relu(self.input_layer[layer_index](inputs))

        for i in range(self.layers - 1):
            z = torch.relu(self.hiddens[i * (3 * self.N + 1) + layer_index](z))

        logits = self.outputs[layer_index](z)
        if len(state.oh_state) > 0:
            dist = torch.softmax(logits, -1)
            return dist
        dist = torch.sigmoid(logits)
        return dist


class MNISTAddModel(PPEBase[MNISTAddState]):

    def __init__(self, args, device='cpu'):
        self.N = args["N"]
        self.device = device

        nrm = NRMMnist(self.N,
                       layers=args["layers"],
                       hidden_size=args["hidden_size"],
                       prune=args["prune"]).to(device)
        super().__init__(nrm,
                         # Perception network
                         MNIST_Net().to(device),
                         amount_samples=args['amt_samples'],
                         belief_size=[10] * 2 * self.N,
                         dirichlet_lr=args['dirichlet_lr'],
                         dirichlet_iters=args['dirichlet_iters'],
                         initial_concentration=args['dirichlet_init'],
                         dirichlet_L2=args['dirichlet_L2'],
                         K_beliefs=args['K_beliefs'],
                         predict_only=args['predict_only'],
                         use_prior=args['use_prior'],
                         nrm_lr=args['nrm_lr'],
                         nrm_loss=args['nrm_loss'],
                         policy=args['policy'],
                         perception_lr=args['perception_lr'],
                         perception_loss=args['perception_loss'],
                         percept_loss_pref=args['percept_loss_pref'],
                         device=device)

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> MNISTAddState:
        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(self.N * 2)]
        y_list = None
        if y is not None:
            y_list = [(torch.div(y,  (10 ** (self.N - i)), rounding_mode='floor') % 10).long() for i in range(self.N + 1)]
        return MNISTAddState(P, self.N, (y_list, w_list), generate_w=generate_w)

    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (batch_size, 2*n)
        """
        stack1 = torch.stack([10 ** (self.N - i - 1) * w[..., i] for i in range(self.N)], -1)
        stack2 = torch.stack([10 ** (self.N - i - 1) * w[..., self.N + i] for i in range(self.N)], -1)

        n1 = stack1.sum(-1)
        n2 = stack2.sum(-1)

        # print(n1, n2, n1 + n2, (n1 + n2).float().mean(-1))

        return n1 + n2

    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        if beam:
            prediction = list(map(lambda syi: syi[:, 0], prediction))
        else:
            y = y.unsqueeze(-1)
        stack = torch.stack([10 ** (self.N - i) * prediction[i] for i in range(self.N + 1)], -1)
        return stack.sum(-1) == y
