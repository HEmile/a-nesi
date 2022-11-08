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
    def __init__(self):
        super().__init__()

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1,
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
            distribution = flow / partition.unsqueeze(-1)

            n_samples = amt_samples if len(flows) == 0 else 1
            if sampler is None:
                action = self.regular_sampler(flow, distribution, n_samples, state)
            else:
                action = sampler(flow, distribution, n_samples, state)
            state = state.next_state(action)

            if len(action.shape) < len(flow.shape):
                action = action.unsqueeze(-1)
            s_flow = flow.gather(-1, action)
            s_dist = distribution.gather(-1, action)
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

        # Product over forward probabilities
        x = torch.stack(result.forward_probabilities, -1).prod(-1)
        # Multiply with partition Z
        x *= result.partitions[0]

        y = result.final_state.success.float()
        if is_wmc:
            # Reward function for weighted model counting weights models by their probability
            y *= result.final_state.log_prob().exp()
        return nn.BCELoss()(x, y)


class NeSyGFlowNet(GFlowNetBase[ST]):

    def __init__(self, mc_gfn: GFlowNetBase[ST], wmc_gfn: GFlowNetBase[ST], greedy_prob: float = 0.2,
                 uniform_prob: float = 0.2, mc_flow_filter=0.01):
        # mc_gfn: Model counting GFlowNet. Might include background knowledge
        # wmc_gfn: Weighted model counting GFlowNet
        super().__init__()
        self.mc_gfn = mc_gfn
        self.wmc_gfn = wmc_gfn
        self.greedy_prob = greedy_prob
        self.uniform_prob = uniform_prob
        self.mc_flow_filter = mc_flow_filter

    def forward(self, state: ST, max_steps: Optional[int] = None, amt_samples=1,
                sampler: Optional[Sampler] = None) -> Tuple[GFlowNetResult[ST], GFlowNetResult[ST]]:
        flows_mc = []
        partitions_mc = []
        forward_probabilities_mc = []

        flows_wmc = []
        partitions_wmc = []
        forward_probabilities_wmc = []

        if sampler is not None:
            print("WARNING: Sampler is ignored in NeSyGFlowNet")
        steps = max_steps
        assert not state.sink and (steps is None or steps > 0)
        while not state.sink and (steps is None or steps > 0):
            # Run the model counting GFlowNet. Sample both uniformly and greedily
            mc_result = self.mc_gfn.forward(state, max_steps=1, amt_samples=amt_samples, sampler=self.mc_sampler)

            # Run the weighted model counting GFlowNet. Sample proportionally, but adjust for impossible actions
            def wmc_flow_adjuster(flow: torch.Tensor, state: ST) -> torch.Tensor:
                # Adjusts the flow such that actions for which there are no models are never sampled
                # TODO: After this step, it's possible that the total flow is 0. This should be handled.
                return flow * (mc_result.final_flow > self.mc_flow_filter).float()
            wmc_result = self.wmc_gfn(state, max_steps=1, amt_samples=amt_samples, flow_pp=wmc_flow_adjuster)

            # Choose which result to use
            mc_prob = self.greedy_prob + self.uniform_prob
            mask = torch.bernoulli(torch.fill(torch.empty_like(mc_result.final_action), mc_prob)).bool()
            action = torch.where(mask, mc_result.final_action, wmc_result.final_action)

            # Update states
            flows_mc.append(mc_result.final_flow.gather(-1, action))
            partitions_mc.append(mc_result.partitions[0])
            forward_probabilities_mc.append(mc_result.final_distribution.gather(-1, action))

            flows_wmc.append(wmc_result.final_flow.gather(-1, action))
            partitions_wmc.append(wmc_result.partitions[0])
            forward_probabilities_wmc.append(wmc_result.final_distribution.gather(-1, action))

            # Move to next state
            state = state.next_state(action)

            if steps:
                steps -= 1

        return GFlowNetResult(state, flows_mc, partitions_mc, forward_probabilities_mc,
                              action, mc_result.final_flow, mc_result.final_distribution), \
               GFlowNetResult(state, flows_wmc, partitions_wmc, forward_probabilities_wmc,
                              action, wmc_result.final_flow, wmc_result.final_distribution)

    def greedy_sampler(self, flow: torch.Tensor, amt_samples: int, state: ST) -> torch.Tensor:
        prior = state.next_prior()
        if len(flow.shape) == 3:
            prior = prior.unsqueeze(1)
        greedy_flow = flow * prior
        greedy_dist = greedy_flow / greedy_flow.sum(-1).unsqueeze(-1)
        sample_shape = (amt_samples,) if amt_samples > 1 else ()
        samples = Categorical(greedy_dist).sample(sample_shape)
        if amt_samples > 1:
            samples = samples.T
        return samples

    def mc_sampler(self, flow: torch.Tensor, distribution: torch.Tensor, amt_samples: int, state: ST) -> torch.Tensor:
        greedy_samples = self.greedy_sampler(flow, amt_samples, state)
        uniform_samples = self.regular_sampler(flow, distribution, amt_samples, state)
        prob_greedy = self.greedy_prob / (self.greedy_prob + self.uniform_prob)
        mask = torch.bernoulli(torch.fill(torch.empty_like(uniform_samples), prob_greedy))
        return mask * greedy_samples + (1 - mask) * uniform_samples

    def flow(self, state: ST) -> torch.Tensor:
        return self.wmc_gfn.flow(state) * (self.mc_gfn.flow(state) > self.mc_flow_filter).float()

    def joint_loss(self, result_mc: GFlowNetResult[ST], result_wmc: GFlowNetResult[ST]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        return self.mc_gfn.loss(result_mc, False), self.wmc_gfn.loss(result_wmc, True)

    def loss(self, result: GFlowNetResult[ST], is_wmc=True) -> torch.Tensor:
        raise NotImplementedError("Use joint_loss instead")
