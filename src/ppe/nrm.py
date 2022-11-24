from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Generic, TypeVar, Callable, Optional

import torch
from torch import nn
from torch.distributions import Categorical
import math

O = TypeVar('O')

class StateBase(ABC, Generic[O]):
    # The difference between y and the constraint: An empty state when doing learning provides the constraint, and
    #  not y. This is to learn the mapping from the probability to y. When doing inference, we provide y
    #  deterministically at the first step.
    constraint: Optional[List[O]] = None
    y: Optional[List[O]] = None
    # True if this state is a sink
    final: bool = False
    # If this state is a sink, this should contain for each sample whether the constraint is satisfied
    success: Optional[torch.Tensor] = None

    def __init__(self, final: bool = False):
        super().__init__()
        self.final = final
        if self.final:
            self.success = self.compute_success()

    @abstractmethod
    def compute_success(self) -> torch.Tensor:
        pass

    @abstractmethod
    def next_state(self, action: torch.Tensor) -> StateBase:
        pass

    @abstractmethod
    def log_p_world(self) -> torch.Tensor:
        # UNconditional log probability of the state (ie, prob of state irrespective of constraint)
        pass

    @abstractmethod
    def next_prior(self) -> torch.Tensor:
        # Conditional probability of the next state given everything in the current state
        pass

    @abstractmethod
    def symbolic_pruner(self) -> torch.Tensor:
        # Uses symbolic reasoning on the current state to figure out which actions are viable. Returns a mask
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
    def __init__(self, lossf='mse-tb', prune=True):
        super().__init__()
        # Saves nonzero results. Shouldn't be used when running the Neurosymbolic GFlowNet as it is guaranteed to sample
        #  models.
        self.lossf = lossf
        self.prune = prune

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1,
                sampler: Optional[Sampler] = None) -> NRMResult[ST]:
        # sampler: If None, this samples in proportion to the flow.
        # Otherwise, this should be a function that takes the flow, the distribution, the number of samples, and the state, and returns a sample

        # Sample (hopefully) positive worlds to estimate gradients
        forward_probabilities = []
        steps = max_steps
        assert not state.final and (steps is None or steps > 0)

        if not sampler:
            sampler = self.regular_sampler

        # The main GFlowNet loop to sample trajectories
        while not state.final and (steps is None or steps > 0):
            distribution = self.distribution(state)
            is_binary = distribution.shape[-1] == 1
            if is_binary:
                distribution = torch.cat([distribution, 1 - distribution], dim=-1)
            if self.prune:
                mask = state.symbolic_pruner().float()
                distribution = distribution * mask
                distribution = distribution / distribution.sum(-1, keepdim=True)

            # TODO: We will also need constraints for w
            if state.constraint is not None and len(state.y) != len(state.constraint):
                action = state.constraint[len(state.y)]
            else:
                # If we have no conditional/constraint, just sample by amount of samples given
                #  Otherwise, we first need to set the conditional (no need to have multiple samples there)
                #  But we also only want to do this once, otherwise we get an exponential explosion of samples
                if state.constraint is None and len(state.y) == 0 or \
                        len(state.w) == 0 and state.constraint is not None and len(state.constraint) == len(state.y):
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
        final_state = state
        return NRMResult(final_state, forward_probabilities, action, s_dist)

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

    def loss(self, result: NRMResult[ST], is_wmc=True) -> torch.Tensor:
        # Implements a modified version of trajectory balance
        # See https://arxiv.org/abs/2201.13259
        # TODO: I should think about how to make this numerically stable...
        # Naive implementation: Compute Z * exp log prob, take as target.
        # This is numerically unstable, as the log prob can be very small, and the partition can be very large.
        # if is_wmc: Weights output rewards by the probability of the sample.
        # If not is_wmc: Rewards are just whether the constraint is satisfied. Used for model counting.

        assert result.final_state.final and result.final_state.success is not None

        log_q = torch.stack(result.forward_probabilities, -1).log().sum(-1)
        # # Why not multiply with partition Z? Because the source node has flow 1!
        # log_x = log_x + result.partitions[0].unsqueeze(-1).log()

        # let's for now assume it's 1... Since we use the perfect sampler.
        y = result.final_state.success.float()

        # For things that are not successes, log_p should be a very small negative number.
        log_p = (1-y) * math.log(1e-8) + result.final_state.log_p_world().detach()

        if self.lossf == 'bce-tb':
            # Product over forward probabilities
            x = torch.exp(log_q)

            if is_wmc:
                # Reward function for weighted model counting weights models by their probability
                y = y * log_p.exp()
            if self.experience_replay:
                pass
            return nn.BCELoss()(x, y)
        elif self.lossf == 'mse-tb':
            return (log_q - log_p).pow(2).mean()
        raise ValueError(f"Unknown loss function {self.lossf}")


class GreedyNRM(NRMBase[ST]):

    def __init__(self, gfn: NRMBase[ST], prune=True, greedy_prob: float = 0.0,
                 uniform_prob: float = 0.0, loss_f='mse-tb'):
        # mc_gfn: Model counting GFlowNet. Might include background knowledge
        # wmc_gfn: Weighted model counting GFlowNet
        super().__init__(loss_f, prune=prune)
        self.gfn = gfn
        self.greedy_prob = greedy_prob
        self.uniform_prob = uniform_prob

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1,
                sampler: Optional[Sampler] = None) -> NRMResult[ST]:
        probs = []

        if sampler is not None:
            print("WARNING: Sampler is ignored in NeSyGFlowNet")
        steps = max_steps
        assert not state.final and (steps is None or steps > 0)
        while not state.final and (steps is None or steps > 0):
            # TODO: This is kinda hacky. This assumes we first deterministically select a constraint, then we start
            #  sampling worlds from there.
            # Run the weighted model counting GFlowNet. Sample proportionally, but prune impossible actions (if self.prune)
            result = self.gfn(state, max_steps=1, amt_samples=amt_samples)

            # Choose which action to use
            mc_prob = self.greedy_prob + self.uniform_prob
            if mc_prob > 0.000001:
                # Sample using background knowledge (mostly runs the greedy sampler)
                # !!This is not used in the current iteration of the framework, and can be ignored for now!!
                mc_action = self.mc_sampler(amt_samples, state)

                mask = torch.bernoulli(torch.fill(torch.empty_like(mc_action, dtype=torch.float), mc_prob)).bool()
                action = torch.where(mask, mc_action, result.final_action)

                # Move to next state
                state = state.next_state(action)
                should_unsqueeze = len(action.shape) < len(result.final_distribution.shape)
                if should_unsqueeze:
                    action = action.unsqueeze(-1)
                # Update states
                p = result.final_distribution.gather(-1, action)

                if len(p.shape) > 2:
                    p = p.squeeze(-1)

                if should_unsqueeze:
                    action = action.squeeze(-1)
            else:
                action = result.final_action
                state = result.final_state
                p = result.final_distribution

            probs.append(p)

            if steps:
                steps -= 1

        return NRMResult(state, probs, action, result.final_distribution)

    def greedy_sampler(self, amount_models: torch.Tensor, amt_samples: int, state: ST) -> torch.Tensor:
        prior = state.next_prior()
        if len(amount_models.shape) == 3:
            prior = prior.unsqueeze(1)
        greedy_flow = amount_models * prior
        greedy_dist = greedy_flow / greedy_flow.sum(-1).unsqueeze(-1)
        sample_shape = (amt_samples,) if amt_samples > 1 else ()
        # TODO: This is instable, because it's possible that the greedy flow is almost 0 everywhere
        #  Or actually I get some nans
        samples = Categorical(greedy_dist).sample(sample_shape)
        if amt_samples > 1:
            samples = samples.T
        return samples

    def mc_sampler(self, amt_samples: int, state: ST) -> torch.Tensor:
        amt_models = state.model_count()
        greedy_samples = None
        if self.greedy_prob > 0:
            # Only use the greedy sampler
            greedy_samples = self.greedy_sampler(amt_models, amt_samples, state)
        if self.uniform_prob > 0:
            total_models = amt_models.sum(-1)
            distribution = amt_models / total_models.unsqueeze(-1)

            uniform_samples = self.regular_sampler(amt_models, distribution, amt_samples, state)
            if self.greedy_prob > 0:
                # Mix the two samplers
                prob_greedy = self.greedy_prob / (self.greedy_prob + self.uniform_prob)
                mask = torch.bernoulli(
                    torch.fill(torch.empty_like(uniform_samples, dtype=torch.float), prob_greedy)).bool()
                chosen_samples = torch.where(mask, greedy_samples, uniform_samples)
                return chosen_samples
            # Only the uniform sampler
            return uniform_samples
        return greedy_samples

    def distribution(self, state: ST) -> torch.Tensor:
        return self.gfn.distribution(state)

    def loss(self, result: NRMResult[ST], is_wmc=True) -> torch.Tensor:
        return self.gfn.loss(result, is_wmc=True)