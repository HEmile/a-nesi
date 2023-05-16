import math
from typing import Optional, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Linear, ReLU, Softmax

from anesi import ANeSIBase
from experiments.MNISTNet import MNIST_Net

from inference_models import StateBase, Constraint, InferenceModelBase, ST, InferenceResult
from util import log_not

EPS = 1E-6


class ViSudoState(StateBase):

    def __init__(self,
                 probability: torch.Tensor,
                 constraint: Constraint,
                 N: int,
                 y: List[Tensor] = [],
                 w: List[Tensor] = [],
                 generate_w: bool = True,
                 final: bool = False):
        # Assuming probability is a b x 2*N x 10 Tensor
        # state: Contains the sampled digits
        # oh_state: Contains the one-hot encoded digits, but _also_ the one-hot encoded value of y
        self.pw = probability

        self.constraint = constraint
        self.y = y
        self.w = w
        self.N = N

        oh_states = []
        self.oh_pw = self.pw.reshape(self.pw.shape[0], -1)
        if len(self.y) > 0:
            if len(self.w) > 0 and len(self.w[0].shape) == 2:
                oh_states.append(self.y[0].unsqueeze(1).expand(-1, self.w[0].shape[1], -1))
                self.oh_pw = self.oh_pw.unsqueeze(1).expand(-1, self.w[0].shape[1], -1)
            else:
                oh_states.append(self.y[0])
        if len(self.w) > 0:
            for wi in self.w:
                oh_states.append(nn.functional.one_hot(wi, N))
        if len(oh_states) > 0:
            self.oh_state = torch.cat(oh_states, dim=-1)

        super().__init__(generate_w, final)

    def log_p_world(self) -> torch.Tensor:
        if self.l_p is not None:
            # Cached log prob
            return self.l_p
        sum = 0.
        for i, d in enumerate(self.w):
            sum += (self.pw[:, i] + EPS).log().gather(1, d)
        self.l_p = sum
        return sum

    def symbolic_pruner(self) -> torch.Tensor:
        if len(self.y) == 0:
            return torch.ones((1, 1, 2), device=self.pw.device)
        if len(self.w) == 0 or not self.y[0].all():
            return torch.ones((1, self.N), device=self.pw.device)
        tot_sample_count = self.w[0].shape[0] * self.w[0].shape[1]
        pruned = torch.ones(tot_sample_count, self.N, device=self.pw.device)
        i = len(self.w)
        x = i // self.N
        y = i % self.N
        _range = torch.arange(0, tot_sample_count, device=self.pw.device, dtype=torch.long)
        obs_indices = set()
        def _set_0s(index, pruned):
            pruned[_range, self.w[index].reshape(-1)] = 0

        for xi in range(x):
            index = xi * self.N + y
            _set_0s(index, pruned)
            obs_indices.add(index)
        for yi in range(y):
            index = x * self.N + yi
            _set_0s(index, pruned)
            obs_indices.add(index)

        sqrt_N = int(math.sqrt(self.N))
        block_x = x // sqrt_N
        block_y = y // sqrt_N
        for xi in range(block_x * sqrt_N, (block_x + 1) * sqrt_N):
            for yi in range(block_y * sqrt_N, (block_y + 1) * sqrt_N):
                index = xi * self.N + yi
                if index in obs_indices or index >= i:
                    continue
                _set_0s(index, pruned)
        pruned = pruned.reshape(self.w[0].shape[0], self.w[0].shape[1], self.N)
        return pruned

    def finished_generating_y(self) -> bool:
        return len(self.y) == 1

    def next_state(self, action: torch.Tensor, beam_selector: Optional[torch.Tensor] = None) -> StateBase:
        if not self.finished_generating_y():
            return ViSudoState(self.pw, self.constraint, self.N, [action], self.w, self.generate_w, False)

        w = self.w
        if beam_selector is not None:
            w = list(map(lambda wi: wi.gather(-1, beam_selector), w))
        return ViSudoState(self.pw, self.constraint, self.N, self.y, w + [action], self.generate_w,
                           len(w) == self.N * self.N - 1)

class ViSudoIMShallow(InferenceModelBase):

    def __init__(self, d1: torch.Tensor, d2: torch.Tensor, groups: torch.Tensor, N: int, hidden_size=50, layers=1, encoding: str = "pair", prune: bool = False):
        super().__init__(prune, "ignore")
        assert encoding in ["pair", "group", "full"]
        self.y_dim = d1.shape[0]
        self.d1 = d1
        self.d2 = d2
        self.groups = groups
        self.N = N
        self.layers = layers

        self.use_groups = encoding in ["group", "full"]

        self.y_in = Linear(2 * N, hidden_size)
        self.group_in = Linear(N * N, hidden_size)
        for i in range(layers - 1):
            setattr(self, f"y_hidden_{i}", Linear(hidden_size, hidden_size))
            setattr(self, f"group_hidden_{i}", Linear(hidden_size, hidden_size))
        self.y_out = Linear(hidden_size, 1)
        self.group_out = Linear(hidden_size, 1)

        self.w_layers = []
        for i in range(N**2):
            modules = [Linear(N**3 + i * N + self.d1.shape[0], hidden_size), ReLU()]
            for j in range(layers - 1):
                modules.append(Linear(hidden_size, hidden_size))
                modules.append(ReLU())
            modules.append(Linear(hidden_size, N))
            modules.append(Softmax(dim=-1))
            self.w_layers.append(torch.nn.Sequential(*modules))

    def distribution(self, state) -> torch.Tensor:
        p1 = state.pw[:, self.d1]
        p2 = state.pw[:, self.d2]
        if self.use_groups:
            pgroups = state.pw[:, self.groups.reshape(-1)]
            pgroups = pgroups.reshape(-1, self.groups.shape[0], self.N * self.N)
        p = torch.cat([p1, p2], dim=2)
        if not state.finished_generating_y():
            z1 = torch.relu(self.y_in(p))
            for i in range(self.layers - 1):
                z1 = torch.relu(getattr(self, f"y_hidden_{i}")(z1))
            dist = torch.sigmoid(self.y_out(z1))

            if self.use_groups:
                z2 = torch.relu(self.group_in(pgroups))
                for i in range(self.layers - 1):
                    z2 = torch.relu(getattr(self, f"group_hidden_{i}")(z2))
                else:
                    dist = torch.cat([dist, torch.sigmoid(self.group_out(z2))], dim=1)
            return dist

        z = torch.cat([state.oh_state, state.oh_pw], dim=-1)
        return self.w_layers[len(state.w)](z)


def compile_comparisons(N: int):
    """
    Compiles what digits need to be compared (need to be different) in a Sudoku puzzle.
    """
    l1 = []
    l2 = []
    groups = []
    sqrt_N = int(math.sqrt(N))
    for i in range(N):
        # Rows
        groups.append([i * N + j for j in range(N)])
        # Columns
        groups.append([j * N + i for j in range(N)])

    for i in range(sqrt_N):
        for j in range(sqrt_N):
            box_group = []
            base = (i * N + j) * sqrt_N
            for x in range(sqrt_N):
                for y in range(sqrt_N):
                    box_group.append(base + x * N + y)
            groups.append(box_group)

    for i in range(N * N):
        xi = i % N
        yi = i // N

        bxi = xi // sqrt_N
        byi = yi // sqrt_N
        for j in range(i + 1, N * N):
            xj = j % N
            yj = j // N

            bxj = xj // sqrt_N
            byj = yj // sqrt_N

            # Same row, same column, same box
            if xi == xj or yi == yj or (bxi == bxj and byi == byj):
                l1.append(i)
                l2.append(j)

    return torch.tensor(l1), torch.tensor(l2), torch.tensor(groups)


class ViSudoModel(ANeSIBase[ViSudoState], nn.Module):

    def __init__(self, args):
        self.N = args["N"]
        self.d1s, self.d2s, self.groups = compile_comparisons(self.N)
        self.encoding = args["encoding"]
        self._test_constraints()

        im = ViSudoIMShallow(self.d1s, self.d2s, self.groups, self.N,
                      hidden_size=args["hidden_size"],
                      layers=args["layers"],
                      encoding=args["encoding"],
                      prune=args["prune"])

        super().__init__(im,
                         # Perception network
                         MNIST_Net(self.N),
                         amount_samples=args['amt_samples'],
                         belief_size=[self.N] * (self.N * self.N),
                         q_lr=args['q_lr'],
                         policy=args['policy'],
                         dirichlet_lr=args['dirichlet_lr'],
                         dirichlet_iters=args['dirichlet_iters'],
                         initial_concentration=args['dirichlet_init'],
                         fixed_alpha=args['fixed_alpha'],
                         dirichlet_L2=args['dirichlet_L2'],
                         K_beliefs=args['K_beliefs'],
                         predict_only=args['predict_only'],
                         P_source=args['P_source'],
                         q_loss=args['q_loss'],
                         perception_lr=args['perception_lr'],
                         perception_loss=args['perception_loss'],
                         percept_loss_pref=args['percept_loss_pref']
                         )
        self.im = im
        self.verbosity = args['verbose']

    def forward(self, preds: torch.Tensor):
        if self.training:
            return self.req_loss(preds)
        raise NotImplementedError()

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> ViSudoState:
        y = [y] if y is not None else []
        # Compute the inverse of torch.stack on w

        w = [w[:, i] for i in range (self.N * self.N)] if w is not None else []
        return ViSudoState(P, (y, w), self.N, generate_w=generate_w)

    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (batch_size, k_vars)
        y: (batch_size, n_clauses)
        """
        d1 = w[..., self.d1s]
        d2 = w[..., self.d2s]
        pairs = (d1 != d2).long()
        if self.encoding == "group":
            groups = w[..., self.groups.reshape(-1)].reshape(w.shape[0], self.groups.shape[0], self.groups.shape[1])
            g_truth = torch.ones(groups.shape[:2], device=groups.device, dtype=torch.long)
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    g_truth *= (groups[:, :, i] != groups[:, :, j]).long()
            return torch.cat([pairs, g_truth], dim=-1)
        return pairs

    def _test_constraints(self):
        if self.N != 9:
            return
        example_puzzle = torch.tensor([
            [4, 3, 5, 2, 6, 9, 7, 8, 1],
            [6, 8, 2, 5, 7, 1, 4, 9, 3],
            [1, 9, 7, 8, 3, 4, 5, 6, 2],
            [8, 2, 6, 1, 9, 5, 3, 4, 7],
            [3, 7, 4, 6, 8, 2, 9, 1, 5],
            [9, 5, 1, 7, 4, 3, 6, 2, 8],
            [5, 1, 9, 3, 2, 6, 8, 7, 4],
            [2, 4, 8, 9, 5, 7, 1, 3, 6],
            [7, 6, 3, 4, 1, 8, 2, 5, 9],
        ])

        constraints = self.symbolic_function(example_puzzle.reshape(1, -1))
        assert torch.min(constraints) == 1

        example_puzzle[4][7] = 2

        constraints = self.symbolic_function(example_puzzle.reshape(1, -1))
        assert torch.min(constraints) == 0

    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        return prediction == y

    def log_q_loss(self, P: torch.Tensor, y: torch.Tensor):
        initial_state = self._one_y_state(P, generate_w=False)
        res = self.im(initial_state)

        log_probs = (res.forward_probabilities[0].squeeze()).log().sum(-1)
        log_probs[y == 0] = log_not(log_probs[y == 0])
        if self.verbosity >= 2:
            self._print_sudoku(P, y, count=1)
        return (-log_probs).mean()

    def _one_y_label(self, batch_size: int, device):
        amt_y = self.d1s.shape[0] + (self.groups.shape[0] if self.encoding == "group" else 0)
        return torch.ones((batch_size, amt_y), device=device, dtype=torch.long)

    def _one_y_state(self, P: torch.Tensor, generate_w=True) -> ViSudoState:
        y = self._one_y_label(P.shape[0], P.device)
        return self.initial_state(P, y, generate_w=generate_w)

    def _print_sudoku(self, P, true_y, count=1):
        w = torch.argmax(P, dim=-1)
        true_puzzle = self.symbolic_function(w).min(-1)[0]
        w = w.view(-1, self.N, self.N)
        succes = true_puzzle == true_y
        for i in range(min(w.shape[0], count)):
            for x in range(self.N):
                if x % math.sqrt(self.N) == 0:
                    print("-----------")
                for y in range(self.N):
                    if y % math.sqrt(self.N) == 0:
                        print(end="|")
                    print(w[i, x, y].item(), end="")
                    if y % math.sqrt(self.N) != math.sqrt(self.N) - 1:
                        print(end=" ")
                print(end="|")
                print()
            print("-----------")
            print()
        print("Succes: ", succes.float().mean())

    def sampled_loss(self, P: torch.Tensor, y: torch.Tensor, compute_perception_loss: bool, compute_q_loss: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        P = P[y == 1]
        y = self._one_y_label(P.shape[0], P.device)
        super().sampled_loss(P, y, compute_perception_loss, compute_q_loss)

    def test(self, x: torch.Tensor, y: torch.Tensor, true_w: Optional[List[torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Algorithm 1
        Sample from the PPE model
        :param x: The data to perform beam search on
        :param y: The true label y to predict and compare to
        :param true_w: The true w (explanation to get to y). Used to evaluate if explanations are correct
        """
        P = self.perception(x)
        if self.verbosity >= 1:
            self._print_sudoku(P, y, count=1)
        initial_state = self._one_y_state(P, generate_w=not self.predict_only)
        result: InferenceResult[ST] = self.q(initial_state)
        prob_y = torch.prod(result.forward_probabilities[0], -1)
        pred_y = (result.forward_probabilities[0] > 0.5).long().min(-1)[0]
        successes = self.success(pred_y, y, beam=False).float()

        prior_predictions = torch.argmax(P, dim=-1)
        prior_y = self.symbolic_function(prior_predictions)
        avg_prior_y = prior_y.float().mean()
        prior_y = prior_y.min(-1)[0]

        successes_prior = (y == prior_y).float().mean()

        # if true_w is not None:
        #     explain_acc = torch.tensor(0., device=successes.device)
        #     if not self.predict_only:
        #         for i in range(len(true_w)):
        #             # Get beam search prediction of w, compare to ground truth w
        #             explain_acc += (result.final_state.w[i][:, 0] == true_w[i]).float().mean()
        #         explain_acc /= len(true_w)
        #
        #     prior_acc = (prior_predictions == torch.stack(true_w, 1)).float().mean()
        #
        #     return torch.mean(successes), successes_prior, explain_acc, prior_acc
        return torch.mean(successes), successes_prior, avg_prior_y, prob_y
