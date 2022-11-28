from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Generic, TypeVar, Callable, Optional

import torch
from torch import nn
from torch.distributions import Categorical

O = TypeVar('O')

class StateBase(ABC, Generic[O]):
    # The difference between y and the constraint: An empty state when doing learning provides the constraint, and
    #  not y. This is to learn the mapping from the probability to y. When doing inference, we provide y
    #  deterministically at the first step.
    constraint: Optional[O] = None
    y: Optional[O] = None
    # True if this state is a sink
    sink: bool = False
    # If this state is a sink, this should contain for each sample whether the constraint is satisfied
    success: Optional[torch.Tensor] = None

    def __init__(self, sink: bool = False):
        super().__init__()
        self.sink = sink
        if self.sink:
            self.success = self.compute_success()

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

    @abstractmethod
    def model_count(self) -> torch.Tensor:
        # Uses symbolic reasoning on the current state to compute for each action how many models there are
        pass

    def viable_actions(self) -> torch.Tensor:
        # Uses symbolic reasoning on the current state to figure out which actions are viable. Returns a mask
        # By default, this counts the amount of models and sees if there are any.
        # Can be overriden if model_count is not implemented
        return self.model_count() > 0


ST = TypeVar("ST", bound=StateBase)


@dataclass
class GFlowNetResult(Generic[ST]):
    final_state: ST
    trajectory_flows: List[torch.Tensor]
    partitions: List[torch.Tensor]
    forward_probabilities: List[torch.Tensor]
    final_action: torch.Tensor
    final_flow: torch.Tensor
    final_distribution: torch.Tensor


Sampler = Callable[[torch.Tensor, torch.Tensor, int, ST], torch.Tensor]
FlowPostProcessor = Callable[[torch.Tensor, ST], torch.Tensor]


class GFlowNetBase(ABC, nn.Module, Generic[ST]):
    def __init__(self, lossf='mse-tb', experience_replay=False):
        super().__init__()
        # Saves nonzero results. Shouldn't be used when running the Neurosymbolic GFlowNet as it is guaranteed to sample
        #  models.
        self.experience_replay = experience_replay
        if experience_replay:
            self.replay_buffer = set[List[ST]]()
        self.lossf = lossf

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1, has_expanded=False,
                sampler: Optional[Sampler] = None, flow_pp: Optional[FlowPostProcessor] = None) -> GFlowNetResult[ST]:
        # sampler: If None, this samples in proportion to the flow.
        # Otherwise, this should be a function that takes the flow, the distribution, the number of samples, and the state, and returns a sample

        # Sample (hopefully) positive worlds to estimate gradients
        flows = []
        partitions = []
        forward_probabilities = []
        steps = max_steps
        assert not state.sink and (steps is None or steps > 0)

        # The main GFlowNet loop to sample trajectories
        while not state.sink and (steps is None or steps > 0):
            flow = self.flow(state)
            if flow_pp is not None:
                flow = flow_pp(flow, state)
            partition = flow.sum(-1)
            # TODO: all flows we use are normalized, I think we can drop this term.
            distribution = flow / partition.unsqueeze(-1)

            if state.constraint is not None and state.y is None:
                action = state.constraint
                # This is the conditional partition!
                partition = flow.mean(-1)
            else:
                # TODO: properly chose amt_samples
                if not has_expanded:
                    n_samples = amt_samples
                    has_expanded = True
                else:
                    n_samples = 1
                if sampler is None:
                    action = self.regular_sampler(flow, distribution, n_samples, state)
                else:
                    action = sampler(flow, distribution, n_samples, state)
            state = state.next_state(action)

            shld_unsqueeze = len(action.shape) < len(flow.shape)
            if shld_unsqueeze:
                action = action.unsqueeze(-1)
            s_flow = flow.gather(-1, action)
            s_dist = distribution.gather(-1, action)
            if shld_unsqueeze:
                action = action.squeeze(-1)
            if len(s_flow) > 2:
                s_flow = s_flow.squeeze(-1)
                s_dist = s_dist.squeeze(-1)
            flows.append(s_flow)
            partitions.append(partition)
            forward_probabilities.append(s_dist)
            if steps:
                steps -= 1
        final_state = state
        return GFlowNetResult(final_state, flows, partitions, forward_probabilities, action, flow, distribution)

    def regular_sampler(self, flow: torch.Tensor, distribution: torch.Tensor, amt_samples: int,
                        state: ST) -> torch.Tensor:
        sample_shape = (amt_samples,) if amt_samples > 1 else ()
        action = Categorical(distribution).sample(sample_shape)
        if amt_samples > 1:
            action = action.T
        return action

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

        log_x = torch.stack(result.forward_probabilities[1:], -1).log().sum(-1)
        # Why not multiply with partition Z? Because the source node has flow 1!
        log_x = log_x + result.partitions[0].unsqueeze(-1).log()

        # let's for now assume it's 1... Since we use the perfect sampler.
        y = result.final_state.success.float()
        log_y = y * result.final_state.log_prob().detach()

        if self.lossf == 'bce-tb':
            # Product over forward probabilities
            x = torch.exp(log_x)

            if is_wmc:
                # Reward function for weighted model counting weights models by their probability
                y = y * log_y.exp()
            if self.experience_replay:
                pass
            return nn.BCELoss()(x, y)
        elif self.lossf == 'mse-tb':
            return (log_x - log_y).pow(2).mean()
        raise ValueError(f"Unknown loss function {self.lossf}")


class NeSyGFlowNet(GFlowNetBase[ST]):

    def __init__(self, gfn: GFlowNetBase[ST], prune=True, greedy_prob: float = 0.2,
                 uniform_prob: float = 0.0, loss_f='mse-tb'):
        # mc_gfn: Model counting GFlowNet. Might include background knowledge
        # wmc_gfn: Weighted model counting GFlowNet
        super().__init__(loss_f)
        self.gfn = gfn
        self.greedy_prob = greedy_prob
        self.uniform_prob = uniform_prob
        self.prune = prune

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1,
                sampler: Optional[Sampler] = None, flow_pp: Optional[FlowPostProcessor] = None) -> GFlowNetResult[ST]:
        flows = []
        partitions = []
        forward_probabilities = []

        if sampler is not None:
            print("WARNING: Sampler is ignored in NeSyGFlowNet")
        steps = max_steps
        assert not state.sink and (steps is None or steps > 0)
        while not state.sink and (steps is None or steps > 0):
            # TODO: This is kinda hacky. This assumes we first deterministically select a constraint, then we start
            #  sampling worlds from there.
            n_samples = amt_samples if len(flows) == 1 else 1

            # Run the weighted model counting GFlowNet. Sample proportionally, but prune impossible actions (if self.prune)
            result = self.gfn(state, max_steps=1, amt_samples=n_samples, flow_pp=self.prune_actions if self.prune else None, has_expanded=len(flows) > 1)

            # Choose which action to use
            mc_prob = self.greedy_prob + self.uniform_prob
            if mc_prob > 0.000001:
                # Sample using background knowledge (mostly runs the greedy sampler)
                mc_action = self.mc_sampler(amt_samples, state)

                mask = torch.bernoulli(torch.fill(torch.empty_like(mc_action, dtype=torch.float), mc_prob)).bool()
                action = torch.where(mask, mc_action, result.final_action)
            else:
                action = result.final_action

            should_unsqueeze = len(action.shape) < len(result.final_flow.shape)
            if should_unsqueeze:
                action = action.unsqueeze(-1)
            # Update states
            flow = result.final_flow.gather(-1, action)
            p = result.final_distribution.gather(-1, action)

            if len(flow.shape) > 2:
                flow = flow.squeeze(-1)
                p = p.squeeze(-1)

            flows.append(flow)
            partitions.append(result.partitions[0])
            forward_probabilities.append(p)

            if should_unsqueeze:
                action = action.squeeze(-1)

            # Move to next state
            state = state.next_state(action)

            if steps:
                steps -= 1

        return GFlowNetResult(state, flows, partitions, forward_probabilities,
                              action, result.final_flow, result.final_distribution)

    def prune_actions(self, flow: torch.Tensor, state: ST) -> torch.Tensor:
        # Adjusts the flow such that actions for which there are no models are never sampled
        return flow * state.viable_actions().float()

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

    def flow(self, state: ST) -> torch.Tensor:
        flow = self.gfn.flow(state)
        if self.prune:
            return self.prune_actions(flow, state)
        return flow

    def loss(self, result: GFlowNetResult[ST], is_wmc=True) -> torch.Tensor:
        return self.gfn.loss(result, is_wmc=True)
