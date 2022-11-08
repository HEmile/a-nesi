from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Generic, TypeVar, Callable, Optional

import torch
from torch import nn
from torch.distributions import Categorical


class StateBase(ABC):
    # True if this state is a sink
    sink: bool = False
    # If this state is a sink, this should contain for each sample whether the constraint is satisfied
    success: Optional[torch.Tensor] = None

    def __init__(self, sink: bool=False):
        super().__init__()
        self.sink = sink
        if self.sink:
            self.compute_success()

    @abstractmethod
    def compute_success(self) -> torch.Tensor:
        pass

    @abstractmethod
    def next_state(self, action: torch.Tensor) -> StateBase:
        pass

    @abstractmethod
    def log_prob(self) -> torch.Tensor:
        # UNconditional log probability of the state (ie, prob of state irrespective of constraint)
        pass

    @abstractmethod
    def next_prior(self) -> torch.Tensor:
        # Conditional probability of the next state given everything in the current state
        pass


ST = TypeVar("ST", bound=StateBase)

@dataclass
class GFlowNetResult(Generic[ST]):
    final_state: ST
    trajectory_flows: List[torch.Tensor]
    partitions: List[torch.Tensor]
    forward_probabilities: List[torch.Tensor]


class GFlowNetBase(ABC, nn.Module, Generic[ST]):
    def __init__(self):
        super().__init__()

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1,
                sampler: Optional[Callable[[torch.Tensor, int, ST], torch.Tensor]] = None) -> GFlowNetResult[ST]:
        # sampler: If None, this samples in proportion to the flow.
        # Otherwise, this should be a function that takes the flow, the number of samples, and the state, and returns a sample

        # Sample (hopefully) positive worlds to estimate gradients
        flows = []
        partitions = []
        forward_probabilities = []
        steps = max_steps
        while not state.sink and (steps is None or steps > 0):
            flow = self.flow(state)
            partition = flow.sum(-1)
            distribution = flow / partition.unsqueeze(-1)

            n_samples = amt_samples if len(flows) == 0 else 1
            if sampler is None:
                action = Categorical(distribution).sample((n_samples,)).T
            else:
                action = sampler(flow, n_samples, state)
            state = state.next_state(action)

            flows.append(flow.gather(-1, action))
            partitions.append(partition)
            forward_probabilities.append(distribution.gather(-1, action))
            if steps:
                steps -= 1
        final_state = state
        return GFlowNetResult(final_state, flows, partitions, forward_probabilities)

    @abstractmethod
    def flow(self, state: ST) -> torch.Tensor:
        pass

    def loss(self, result: GFlowNetResult[ST], is_wmc=True) -> torch.Tensor:
        # Implements a modified version of trajectory balance
        # See https://arxiv.org/abs/2201.13259
        # TODO: I should think about how to make this numerically stable...
        # Naive implementation: Compute Z * exp log prob, take as target.
        # This is numerically unstable, as the log prob can be very small, and the partition can be very large.
        # if is_wmc: Weights output rewards by the probability of the sample.
        # If not is_wmc: Rewards are just whether the constraint is satisfied. Used for model counting.

        assert result.final_state.sink and result.final_state.success is not None

        # Product over forward probabilities
        x = torch.stack(result.forward_probabilities, -1).prod(-1)
        # Multiply with partition Z
        x *= result.partitions[0]

        y = result.final_state.success.float()
        if is_wmc:
            # Reward function for weighted model counting weights models by their probability
            y *= result.final_state.log_prob().exp()
        return nn.BCELoss()(x, y)


class GFlowNetProb(GFlowNetBase):
    counter_gfn: GFlowNetBase

    def __init__(self, counter_gfn: GFlowNetBase, stable_prob: float = 0.1):
        super().__init__()
        self.counter_gfn = counter_gfn
        self.stable_prob = stable_prob

    def flow(self, state: ST) -> torch.Tensor:
        pass
