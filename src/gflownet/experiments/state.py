from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from gflownet.gflownet import StateBase

StateRep = List[Tensor]

EPS = 1E-6

class MNISTAddState(StateBase):

    l_p: Optional[Tensor] = None

    def __init__(self, probability: torch.Tensor, N: int, constraint: Optional[Tensor] = None,
                 oh_query: Optional[Tensor] = None, state: List[Tensor] = [], oh_state: List[Tensor] = [],
                 expanded_pw: Optional[Tensor] = None, sink: bool = False):
        # Assuming probability is a b x 2*N x 10 Tensor
        assert 2 * N >= len(state)
        self.N = N
        self.pw = probability
        self.expanded_pw = expanded_pw
        self.constraint = constraint
        self.oh_query = oh_query
        if self.oh_query is None:
            self.oh_query = one_hot(constraint, self.n_classes()).float()
        if len(state) == 1:
            self.oh_query = self.oh_query.unsqueeze(1).expand(-1, state[0].shape[1], -1)
            self.expanded_pw = self.pw.flatten(1).unsqueeze(1).expand(-1, state[0].shape[1], -1)
        self.state = state
        if len(state) != len(oh_state):
            raise ValueError("state and oh_state must have the same length")
        self.oh_state = oh_state

        super().__init__(sink)

    def next_state(self, action: torch.Tensor) -> MNISTAddState:
        assert not self.sink
        assert len(self.state) < 2 * self.N

        state = self.state + [action]
        oh_state = self.oh_state + [one_hot(action, 10).float()]

        sink = len(state) == 2 * self.N
        return MNISTAddState(self.pw, self.N, self.constraint, self.oh_query, state, oh_state, self.expanded_pw, sink)

    def compute_success(self) -> torch.Tensor:
        assert self.state is not None
        assert self.constraint is not None
        assert self.sink
        assert len(self.state) == 2 * self.N
        # Compute constraint, ie whether the sum of the numbers is equal to the query
        stack1 = torch.stack([10 ** (self.N - i - 1) * self.state[:self.N][i] for i in range(self.N)], -1)
        stack2 = torch.stack([10 ** (self.N - i - 1) * self.state[self.N:][i] for i in range(self.N)], -1)
        n1 = stack1.sum(-1)
        n2 = stack2.sum(-1)
        return n1 + n2 == self.constraint.unsqueeze(-1)

    def n_classes(self) -> int:
        return 2 * (10 ** self.N) - 1

    def log_prob(self) -> torch.Tensor:
        if self.l_p is not None:
            # Cached log prob
            return self.l_p
        sum = 0.
        for i, d in enumerate(self.state):
            sum += (self.pw[:, i] + EPS).log().gather(1, d)
        self.l_p = sum
        return sum

    def next_prior(self) -> torch.Tensor:
        assert not self.sink
        if self.constraint is None:
            # TODO
            raise NotImplementedError()
        return self.pw[:, len(self.state)]

    def probability_vector(self) -> torch.Tensor:
        if self.expanded_pw is not None:
            return self.expanded_pw
        return self.pw.flatten(1)

    def model_count(self) -> torch.Tensor:
        if self.N == 1:
            # We'll do other cases later

            if self.constraint is None:
                # Should return the amount of models for each query
                raise NotImplementedError()

            if len(self.state) == 0:
                query = self.constraint.unsqueeze(-1)
                rang = torch.arange(10, device=query.device).unsqueeze(0)
                first_comp = rang <= query
                second_comp = rang >= query - 9
                return torch.logical_and(first_comp, second_comp).int()
            else:
                d2 = self.constraint.unsqueeze(-1) - self.state[0]
                # TODO: This assumes for every d1 there is a solution (ie 0 <= constraint - d1 <= 9)
                return torch.nn.functional.one_hot(d2, 10).int()
        raise NotImplementedError()
