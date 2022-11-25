from __future__ import annotations
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from nrm import StateBase, Constraint

StateRep = List[Tensor]

EPS = 1E-6

class MNISTAddState(StateBase):

    l_p: Optional[Tensor] = None

    def __init__(self, probability: torch.Tensor, N: int, constraint: Constraint,
                 y: List[Tensor] = [], w: List[Tensor] = [], oh_state: List[Tensor] = [],
                 expanded_pw: Optional[Tensor] = None, generate_w: bool=True, final: bool = False):
        # Assuming probability is a b x 2*N x 10 Tensor
        # state: Contains the sampled digits
        # oh_state: Contains the one-hot encoded digits, but _also_ the one-hot encoded value of y
        assert 2 * N >= len(w)
        self.N = N
        self.pw = probability
        self.expanded_pw = expanded_pw

        self.constraint = constraint
        self.y = y
        self.w = w
        self.oh_state = oh_state

        if len(w) + len(y) != len(oh_state):
            raise ValueError("oh_state must have the same length as the w and y lists")

        super().__init__(generate_w, final)

    def next_state(self, action: torch.Tensor) -> MNISTAddState:
        assert not self.final
        assert len(self.w) < 2 * self.N

        y = self.y
        w = self.w
        oh_state = self.oh_state
        expanded_pw = self.expanded_pw
        if len(y) < self.N + 1:
            y = y + [action]
            if len(y) == 1:
                # Binary RVs don't need one-hot.
                oh_state = oh_state + [action.float().unsqueeze(-1)]
            else:
                oh_state = oh_state + [one_hot(action, 10).float()]
        else:
            w = w + [action]
            oh_state = oh_state + [one_hot(action, 10).float()]

        # # If we sample multiple samples, then if we deterministically choose using the constraint, it is missing
        # #  the sample dimension. This expands those tensors to include the sample dimension for future computation.
        # if len(oh_state) == len(self.constraint) + 1:
        #     expanded_pw = self.pw.flatten(1).unsqueeze(1).expand(-1, w[0].shape[1], -1)
        #     oh_state[:-1] = map(lambda s: s.unsqueeze(1).expand(-1, w[0].shape[1], -1), oh_state[:-1])

        final = (len(w) == 2 * self.N)

        return MNISTAddState(self.pw, self.N, self.constraint, y, w, oh_state, expanded_pw, generate_w=self.generate_w,
                             final=final)

    def compute_success(self) -> torch.Tensor:
        assert self.w is not None
        assert self.y is not None
        assert self.final
        assert len(self.w) == 2 * self.N
        assert len(self.y) == self.N + 1
        # Compute constraint, ie whether the sum of the numbers is equal to the number represented by y
        stack1 = torch.stack([10 ** (self.N - i - 1) * self.w[:self.N][i] for i in range(self.N)], -1)
        stack2 = torch.stack([10 ** (self.N - i - 1) * self.w[self.N:][i] for i in range(self.N)], -1)

        n1 = stack1.sum(-1)
        n2 = stack2.sum(-1)

        ny = self.query_to_number().unsqueeze(-1)

        return n1 + n2 == ny

    def query_to_number(self) -> torch.Tensor:
        stacky = torch.stack([10 ** (self.N - i) * self.y[i] for i in range(self.N + 1)], -1)
        return stacky.sum(-1)

    def log_p_world(self) -> torch.Tensor:
        if self.l_p is not None:
            # Cached log prob
            return self.l_p
        sum = 0.
        for i, d in enumerate(self.w):
            sum += (self.pw[:, i] + EPS).log().gather(1, d)
        self.l_p = sum
        return sum

    def next_prior(self) -> torch.Tensor:
        assert not self.final
        if self.y is None:
            if self.constraint is not None:
                return one_hot(self.constraint)
            model_count = self.model_count()
            return model_count / model_count.sum(-1, keepdim=True)
        return self.pw[:, len(self.w)]

    def probability_vector(self) -> torch.Tensor:
        if self.expanded_pw is not None:
            return self.expanded_pw
        return self.pw.flatten(1)

    def _amount_models_y(self, y: torch.Tensor) -> torch.Tensor:
        return 10 ** self.N - torch.abs(y - (10 ** self.N - 1))

    def symbolic_pruner(self) -> torch.Tensor:
        if len(self.y) < self.N + 1:
            # if self.constraint is not None:
            #     # Should return the amount of models for each query
            #     # return one_hot(self.constraint, self.n_classes()) * self._amount_models_y(self.constraint).unsqueeze(-1)
            #     return torch.ones_like(self.constraint).unsqueeze(-1).expand(-1, self.n_classes())
            # TODO: This isn't an actual model count but I couldn't be bothered lol
            if len(self.y) == 0:
                return torch.ones((2,)).unsqueeze(0)
            onez = torch.ones((10,))
            if len(self.y) == self.N:
                onez_without_nine = torch.ones_like(onez)
                onez_without_nine[-1] = 0
                return (torch.outer(self.y[0], onez_without_nine) + torch.outer(1 - self.y[0], onez))
            return onez
            # return self._amount_models_y(torch.arange(1, self.n_classes()).unsqueeze(0))
        if self.N == 1:
            # We'll do other cases later
            ny = self.query_to_number()
            if len(self.w) == 0:
                rang = torch.arange(10, device=ny.device).unsqueeze(0)
                ny = ny.unsqueeze(-1)
                first_comp = rang <= ny
                second_comp = rang >= ny - 9
                return torch.logical_and(first_comp, second_comp).int()
            else:
                d2 = ny - self.w[0]
                # TODO: This assumes for every d1 there is a solution (ie 0 <= constraint - d1 <= 9)
                return torch.nn.functional.one_hot(d2, 10).int()
        raise NotImplementedError()

    def finished_generating_y(self) -> bool:
        return len(self.y) == self.N + 1
