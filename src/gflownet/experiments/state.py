from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from gflownet.gflownet import StateBase

StateRep = Tuple[List[Tensor], List[Tensor]]

EPS = 1E-6

class MNISTAddState(StateBase):

    l_p: Optional[Tensor] = None

    def __init__(self, probability: StateRep, N: int, constraint: Optional[Tensor] = None, state: StateRep = ([], []),
                 sink: bool = False):
        super().__init__(sink)
        self.probability = probability
        self.constraint = constraint
        self.state = state
        self.N = N
        d1s, d2s = self.state
        assert N >= len(d1s) >= len(d2s)

    def next_state(self, action: torch.Tensor) -> MNISTAddState:
        assert not self.sink
        d1s, d2s = self.state
        assert len(d1s) < self.N or (len(d2s) < self.N and len(d1s) == self.N)

        if len(d1s) < self.N:
            d1s = d1s + [action]
        else:
            d2s = d2s + [action]

        sink = len(d1s) == self.N and len(d2s) == self.N
        return MNISTAddState(self.probability, self.N, self.constraint, (d1s, d2s), sink)

    def compute_success(self) -> torch.Tensor:
        assert self.state is not None
        assert self.constraint is not None
        assert self.sink
        d1s, d2s = self.state
        assert len(d1s) == len(d2s) == self.N
        # Compute constraint, ie whether the sum of the numbers is equal to the query
        n1 = torch.stack([10 ** (self.N - i) * d1s[i] for i in range(self.N)]).sum(-1)
        n2 = torch.stack([10 ** (self.N - i) * d2s[i] for i in range(self.N)]).sum(-1)
        return n1 + n2 == self.constraint

    def n_classes(self) -> int:
        return 2 * (10 ** self.N) - 1

    def log_prob(self) -> torch.Tensor:
        if self.l_p is not None:
            # Cached log prob
            return self.l_p
        sum = 0.
        for d, p in zip(self.state[0]+self.state[1], self.probability[0]+self.probability[1]):
            d = d.T
            sum += (p + EPS).log().gather(1, d)
        self.l_p = sum
        return sum
