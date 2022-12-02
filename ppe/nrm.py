from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Generic, TypeVar, Callable, Optional
import torch
from torch import nn
from torch.distributions import Categorical

O = TypeVar('O')
W = TypeVar('W')

Constraint = Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]


class StateBase(ABC):
    # The difference between y and the constraint: An empty state when doing learning provides the constraint, and
    #  not y. This is to learn the mapping from the probability to y. When doing inference, we provide y
    #  deterministically at the first step.
    constraint: Constraint = (None, None)
    y: Optional[List[torch.Tensor]] = None
    w: Optional[List[torch.Tensor]] = None
    # True if this state is final (completely generated)
    final: bool
    generate_w: bool

    def __init__(self, generate_w: bool=True, final: bool = False):
        super().__init__()
        self.final = final
        self.generate_w = generate_w

    @abstractmethod
    def next_state(self, action: torch.Tensor, beam_selector: Optional[torch.Tensor] = None) -> StateBase:
        """
        :param action: The action taken. Tensor of shape (batch, amount_samples)
        :param beam_selector: The previous action picked during beam search. Tensor of shape (batch, amount_samples)
          Set to None if not using beam search (when sampling).
        :return: The next state
        """
        pass

    @abstractmethod
    def log_p_world(self) -> torch.Tensor:
        # UNconditional log probability of the state (ie, prob of state irrespective of constraint)
        pass

    @abstractmethod
    def symbolic_pruner(self) -> torch.Tensor:
        # Uses symbolic reasoning on the current state to figure out which actions are viable. Returns a mask
        pass

    @abstractmethod
    def finished_generating_y(self) -> bool:
        pass


ST = TypeVar("ST", bound=StateBase)


@dataclass
class NRMResult(Generic[ST]):
    final_state: ST
    forward_probabilities: List[torch.Tensor]
    final_action: torch.Tensor
    final_distribution: torch.Tensor


Sampler = Callable[[torch.Tensor, int, ST], torch.Tensor]


class NRMBase(ABC, nn.Module, Generic[ST]):
    def __init__(self, prune=True):
        super().__init__()
        # Saves nonzero results. Shouldn't be used when running the Neurosymbolic GFlowNet as it is guaranteed to sample
        #  models.
        self.prune = prune

    def _compute_distribution(self, state: ST) -> torch.Tensor:
        _distribution = self.distribution(state)
        is_binary = _distribution.shape[-1] == 1
        if is_binary:
            _distribution = torch.cat([_distribution, 1 - _distribution], dim=-1)
        if self.prune:
            mask = state.symbolic_pruner().float()
            assert not (mask == 0).all(dim=-1).any()

            distribution = (_distribution + 10e-15) * mask
            distribution = distribution / (distribution.sum(-1, keepdim=True))
        else:
            distribution = _distribution
        return distribution

    def forward(self,
                state: ST,
                max_steps: Optional[int] = None,
                amt_samples=1,
                sampler: Optional[Sampler] = None) -> NRMResult[ST]:
        # sampler: If None, this samples in proportion to the flow.
        # Otherwise, this should be a function that takes the flow, the distribution, the number of samples, and the state, and returns a sample

        # Sample (hopefully) positive worlds to estimate gradients
        forward_probabilities = []
        steps = max_steps
        assert not state.final and (steps is None or steps > 0)

        if not sampler:
            sampler = self.regular_sampler

        while not state.final and (steps is None or steps > 0):
            distribution = self._compute_distribution(state)

            constraint_y = state.constraint[0]
            constraint_w = state.constraint[1]

            if constraint_y is not None and len(state.y) < len(constraint_y):
                action = constraint_y[len(state.y)]
            elif constraint_w is not None and len(state.w) < len(constraint_w):
                action = constraint_w[len(state.w)]
            else:
                # If we have no conditional/constraint, just sample by amount of samples given
                #  Otherwise, we first need to set the conditional (no need to have multiple samples there)
                #  But we also only want to do this once, otherwise we get an exponential explosion of samples
                if state.constraint == (None, None) and len(state.y) == 0 or \
                        len(state.w) == 0 and state.constraint[0] != None \
                        and state.constraint[1] == None and len(state.constraint[0]) == len(state.y):
                    n_samples = amt_samples
                else:
                    n_samples = 1

                action = sampler(distribution, n_samples, state)
            state = state.next_state(action)

            shld_unsqueeze = len(action.shape) < len(distribution.shape)
            if shld_unsqueeze:
                action = action.unsqueeze(-1)

            s_dist = distribution.gather(-1, action)

            if shld_unsqueeze:
                action = action.squeeze(-1)
            if len(s_dist) > 2:
                s_dist = s_dist.squeeze(-1)
            forward_probabilities.append(s_dist)
            if steps:
                steps -= 1

            if not state.generate_w and state.finished_generating_y():
                break
        final_state = state
        return NRMResult(final_state, forward_probabilities, action, s_dist)

    def beam(self, initial_state: ST, beam_size: int):
        """
        Beam search on the NRM model given initial state
        """
        # List of [batch, beam_size]
        forward_probabilities = []

        state = initial_state
        first_iteration = True
        while not state.final:
            distribution = self._compute_distribution(state)

            if first_iteration:
                # [batch, amt_actions]
                k = min(beam_size, distribution.shape[-1])
                probs, action = torch.topk(distribution, k, dim=-1)
                forward_probabilities.append(probs)
                state = state.next_state(action)
                first_iteration = False
            else:
                # distribution: [batch, cur_beam_size, amt_actions]
                log_probs = torch.stack(forward_probabilities, dim=-1).log().sum(-1)
                log_probs = log_probs.unsqueeze(-1) + distribution.log()

                # [batch, cur_beam_size * amt_actions]
                log_probs = log_probs.reshape(log_probs.shape[0], -1)
                # Define how many actions we pick. If we have less than the beam size, we pick all of them.
                #  However, some actions will have probability 0 (symbolic pruner), so we need to filter those out.
                #  Those will be at the bottom of the ordering, so we just need to restrict the size of k
                k = min(beam_size, log_probs.shape[-1] - torch.isinf(log_probs).sum(-1).max().item())
                # [batch, k]
                _, action_flat = torch.topk(log_probs, k, dim=-1)

                prev_action = torch.div(action_flat, distribution.shape[-1], rounding_mode='floor')
                action = action_flat % distribution.shape[-1]

                for i in range(len(forward_probabilities)):
                    forward_probabilities[i] = forward_probabilities[i].gather(-1, prev_action)
                dist_flat = distribution.reshape(distribution.shape[0], -1)
                forward_probabilities.append(dist_flat.gather(-1, action_flat))

                state = state.next_state(action, prev_action)

            if not state.generate_w and state.finished_generating_y():
                break
        final_state = state
        return NRMResult(final_state, forward_probabilities, action, distribution)

    def regular_sampler(self, distribution: torch.Tensor, amt_samples: int,
                        state: ST) -> torch.Tensor:
        sample_shape = (amt_samples,) if amt_samples > 1 else ()
        action = Categorical(distribution).sample(sample_shape)
        if amt_samples > 1:
            action = action.T
        return action

    @abstractmethod
    def distribution(self, state: ST) -> torch.Tensor:
        pass
