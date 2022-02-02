from pyswip import Atom
from typing import TYPE_CHECKING, Callable, Sequence, Dict, List

import random

from distutils.dist import Distribution

import torch
from problog.logic import Term
from torch.distributions import OneHotCategorical

from deepproblog.query import Query
from storch import StochasticTensor, storch

if TYPE_CHECKING:
    from deepproblog.model import Model
    from deepproblog.sampling.sample import MethodFactory


class Sampler(torch.nn.Module):
    model: "Model"
    network_name: str
    def __init__(self, method_factory: "MethodFactory", n: int):
        super().__init__()
        self.method_factory: "MethodFactory" = method_factory
        self.n = n

    def sample(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
        pass

    def __call__(self, queries: Sequence[Query]):
        self.prepare_sampler(queries)

    def update_sampler(self, found_results: torch.Tensor):
        pass

    def parents(self) -> List[storch.Tensor]:
        pass

    def is_batched(self) -> bool:
        return True

    def get_hyperparameters(self) -> dict:
        return {
            "n": self.n,
        }

class IndependentSampler(Sampler):

    # Just samples according to the probability of the NN distribution
    # Samples from all distributions independently
    def get_sample(self, term: Term) -> StochasticTensor:
        to_evaluate = [(term.args[0], term.args[1])]
        # Compute probabilities by evaluating model
        res: torch.Tensor = self.model.evaluate_nn(to_evaluate)[to_evaluate[0]]
        distr: Distribution = OneHotCategorical(res)
        if random.randint(0, 20) == 1:
            print("Digit 1")
            print(res)
        if random.randint(0, 20) == 1:
            print("Digit 2")
            print(res)

        # Sample using Storchastic
        method = self.method_factory("z", self.n)
        return method.sample(distr)

    def is_batched(self) -> bool:
        return False

    def get_hyperparameters(self) -> dict:
        hps = super().get_hyperparameters()
        hps["type"] = "independent"
        return hps