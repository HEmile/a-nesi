from typing import Protocol, Optional

import torch
import storch

from storch import CostTensor, StochasticTensor
from storch.method import Baseline, Method, ScoreFunction, RaoBlackwellSF
from storch.method.baseline import BatchAverageBaseline
from storch.sampling import SamplingMethod


class BaselineSampler(Baseline):
    # TODO: Assign this one to sampler instead of BaselineLearner
    def __init__(self):
        super().__init__()
        self.batch_avg_baseline = BatchAverageBaseline()
        # Const baseline of -0.1 since we want to punish whenever it samples all 0s
        # self.const_all_zeros = torch.tensor(-0.1)

    def compute_baseline(
            self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> torch.Tensor:
        # TODO: Assumes the count over amt samples is z
        cond_any_ones = (storch.sum(cost_node, "z") < 0).float()
        avg_baseline = self.batch_avg_baseline.compute_baseline(tensor, cost_node)
        return avg_baseline * cond_any_ones + (1 / tensor.n) * (1 - cond_any_ones)

class BaselineLearner(Baseline):
    def __init__(self):
        super().__init__()
        self.batch_avg_baseline = BatchAverageBaseline()
        # Const baseline of -0.1 since we want to punish whenever it samples all 0s
        # self.const_all_zeros = torch.tensor(-0.1)

    def compute_baseline(
            self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> torch.Tensor:
        # TODO: Assumes the count over amt samples is z
        # TODO: Is this a valid baseline? It takes into account its own value.
        cond_any_ones = (storch.sum(cost_node, "z") < 0).float()
        cond_any_zeros = (storch.sum(cost_node == torch.zeros(cost_node.size()), "z") > 0).float()
        avg_baseline = self.batch_avg_baseline.compute_baseline(tensor, cost_node)
        return avg_baseline * cond_any_ones * cond_any_zeros + \
               (-1 / tensor.n) * (1 - cond_any_ones) + \
               -(1 - 1 / tensor.n) * (1 - cond_any_zeros)


# Super cool type hinting trix https://stackoverflow.com/questions/68386130/how-to-type-hint-a-callable-of-a-function-with-default-arguments
class MethodFactory(Protocol):
    def __call__(self, name: str, amt_samples: int, sampling_method: Optional[SamplingMethod]) -> Method:
        ...


def factory_storch_method(name="hybrid-baseline") -> MethodFactory:
    seq_estim = None
    def create_storch_method(atom_name: str, amt_samples: int, sampling_method: Optional[SamplingMethod]=None):
        nonlocal seq_estim
        if name == 'hybrid-baseline':
            return ScoreFunction(atom_name, n_samples=amt_samples, sampling_method=sampling_method,
                                 baseline_factory=lambda s, c: BaselineLearner())
        if name == 'rao-blackwell':
            # Seq-based estimators share the same estimator object at each step.
            if not seq_estim:
                seq_estim = RaoBlackwellSF(atom_name, amt_samples)
            return seq_estim
        if name == 'vanilla-sf':
            return ScoreFunction(atom_name, n_samples=amt_samples, sampling_method=sampling_method,
                                 baseline_factory=None)
        return ScoreFunction(atom_name, n_samples=amt_samples, sampling_method=sampling_method,
                             baseline_factory='batch-average')

    return create_storch_method
