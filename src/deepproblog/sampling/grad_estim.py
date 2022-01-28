from typing import Protocol, Optional

import torch
import storch

from storch import CostTensor, StochasticTensor
from storch.method import Baseline, Method, ScoreFunction
from storch.method.baseline import BatchAverageBaseline
from storch.sampling import SamplingMethod


class HybridBaseline(Baseline):
    def __init__(self):
        super().__init__()
        self.batch_avg_baseline = BatchAverageBaseline()
        # Const baseline of -0.1 since we want to punish whenever it samples all 0s
        self.const = torch.tensor(-0.1)

    def compute_baseline(
            self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> torch.Tensor:
        # TODO: Assumes the count over amt samples is z
        cond = (storch.sum(cost_node, "z") < 0).float()
        avg_baseline = self.batch_avg_baseline.compute_baseline(tensor, cost_node)
        return avg_baseline * cond + self.const * (1-cond)


# Super cool type hinting trix https://stackoverflow.com/questions/68386130/how-to-type-hint-a-callable-of-a-function-with-default-arguments
class MethodFactory(Protocol):
    def __call__(self, name: str, amt_samples: int, sampling_method: Optional[SamplingMethod]) -> Method:
        ...


def factory_storch_method(name="hybrid-baseline") -> MethodFactory:
    def create_storch_method(atom_name: str, amt_samples: int, sampling_method: Optional[SamplingMethod]=None):
        if name == 'hybrid-baseline':
            return ScoreFunction(atom_name, n_samples=amt_samples, sampling_method=sampling_method,
                                 baseline_factory=lambda s, c: HybridBaseline())
        return ScoreFunction(atom_name, n_samples=amt_samples, sampling_method=sampling_method,
                             baseline_factory='batch-average')

    return create_storch_method
