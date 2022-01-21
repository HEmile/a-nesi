from typing import TYPE_CHECKING

import random

from distutils.dist import Distribution

import torch
from problog.logic import Term
from torch.distributions import OneHotCategorical

from deepproblog.query import Query
from storch import StochasticTensor
from storch.method import Method
if TYPE_CHECKING:
    from deepproblog.model import Model


class Sampler(object):
    def __init__(self, model: "Model"):
        self.model = model

    def prepare_sampler(self, query: Query):
        pass

    def get_sample(self, term: Term, method: Method) -> StochasticTensor:
        pass

    def update_sampler(self, found_results: torch.Tensor):
        pass

class DefaultSampler(Sampler):
    # Just samples according to the probability of the NN distribution
    def get_sample(self, term: Term, method: Method) -> StochasticTensor:
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
        return method.sample(distr)