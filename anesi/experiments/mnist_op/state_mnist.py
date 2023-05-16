from __future__ import annotations
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from inference_models import StateBase, Constraint

StateRep = List[Tensor]

EPS = 1E-6

class MNISTState(StateBase):

    l_p: Optional[Tensor] = None

    def __init__(self,
                 probability: torch.Tensor,
                 N: int,
                 constraint: Constraint,
                 y_dims: List[int],
                 y: List[Tensor] = [],
                 w: List[Tensor] = [],
                 oh_state: List[Tensor] = [],
                 expanded_pw: Optional[Tensor] = None,
                 generate_w: bool=True,
                 final: bool = False):
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

        self.y_dims = y_dims

        if len(w) + len(y) != len(oh_state):
            raise ValueError("oh_state must have the same length as the w and y lists")

        super().__init__(generate_w, final)

    def next_state(self, action: torch.Tensor, beam_selector: Optional[torch.Tensor] = None) -> MNISTState:
        assert not self.final
        assert len(self.w) < 2 * self.N

        y = self.y
        w = self.w
        oh_state = self.oh_state
        expanded_pw = self.expanded_pw

        if beam_selector is not None:
            y = list(map(lambda yi: yi.gather(-1, beam_selector), y))
            w = list(map(lambda wi: wi.gather(-1, beam_selector), w))
            oh_state = list(map(lambda ohi: ohi.gather(1, beam_selector.unsqueeze(-1).expand((-1, -1, ohi.shape[-1]))), oh_state))
            expanded_pw = None

        if len(y) < len(self.y_dims):
            y_dim_i = self.y_dims[len(y)]
            y = y + [action]
            if y_dim_i == 1:
                # Binary RVs don't need one-hot.
                oh_state = oh_state + [action.float().unsqueeze(-1)]
            else:
                oh_state = oh_state + [one_hot(action, y_dim_i).float()]
        else:
            w = w + [action]
            oh_state = oh_state + [one_hot(action, 10).float()]

        # # If we sample multiple samples, then if we deterministically choose using the constraint, it is missing
        # #  the sample dimension. This expands those tensors to include the sample dimension for future computation.
        if len(oh_state[-1].shape) == 3 and len(oh_state[0].shape) == 2:
            oh_state[:-1] = map(lambda s: s.unsqueeze(1).expand(-1, w[0].shape[1], -1), oh_state[:-1])

        final = (len(w) == 2 * self.N)

        return self.__class__(self.pw,
                             self.N,
                             self.constraint,
                             self.y_dims,
                             y,
                             w,
                             oh_state,
                             expanded_pw,
                             generate_w=self.generate_w,
                             final=final)

    def query_to_number(self) -> torch.Tensor:
        stacky = torch.stack([10 ** (self.N - i) * self.y[i] for i in range(len(self.y_dims))], -1)
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

    def probability_vector(self) -> torch.Tensor:
        if len(self.oh_state) > 0 and len(self.oh_state[-1].shape) == 3:
            if self.expanded_pw is None:
                self.expanded_pw = self.pw.flatten(1).unsqueeze(1).expand(-1, self.oh_state[-1].shape[1], -1)
            return self.expanded_pw
        return self.pw.flatten(1)

    def symbolic_pruner(self) -> torch.Tensor:
        raise NotImplementedError()

    def finished_generating_y(self) -> bool:
        return len(self.y) == len(self.y_dims)
