from typing import TYPE_CHECKING, Callable

import random

from distutils.dist import Distribution

import torch
from problog.logic import Term
from torch.distributions import OneHotCategorical

from deepproblog.query import Query
from storch import StochasticTensor

if TYPE_CHECKING:
    from deepproblog.model import Model
    from deepproblog.sampling.sample import MethodFactory


class Sampler(torch.nn.Module):
    def __init__(self, model: "Model", method_factory: "MethodFactory", n: int):
        super().__init__()
        self.model = model
        self.method_factory: "MethodFactory" = method_factory
        self.n = n

    def prepare_sampler(self, query: Query):
        pass

    def __call__(self, query: Query):
        self.prepare_sampler(query)

    def get_sample(self, term: Term) -> StochasticTensor:
        pass

    def update_sampler(self, found_results: torch.Tensor):
        pass

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