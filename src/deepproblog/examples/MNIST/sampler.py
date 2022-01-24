from typing import Callable

import torch
from problog.logic import Term
from torch.distributions import OneHotCategorical

from deepproblog.examples.MNIST.network import MNIST_NETWORK_NAME
from deepproblog.model import Model
from deepproblog.query import Query
from deepproblog.sampling.grad_estim import MethodFactory
from deepproblog.sampling.sampler import Sampler
from storch import StochasticTensor
from storch.method import Method
import torch.nn as nn
import storch
from torch.nn.functional import one_hot


class AdditionSampler(Sampler):
    embedding: torch.Tensor
    sample1: storch.StochasticTensor

    def __init__(self, model: Model, method: MethodFactory, n: int):
        super().__init__(model, method, n)
        self.max_classes = 19
        self.lin1 = nn.Linear(2 * 10 + self.max_classes, 30)
        self.outp1 = nn.Linear(30, 10)
        self.outp2 = nn.Linear(30 + 10, 10)
        self.sample1 = None

    def prepare_sampler(self, query: Query):
        # TODO: Do this in batches of queries.
        #  When implementing that, pass it as a list into evaluate_nn, as it will automatically batch things.
        query_s = query.substitute()
        arg1 = Term(".", query_s.query.args[0], Term('[]'))
        nn_query_1 = (MNIST_NETWORK_NAME, arg1)
        arg2 = Term(".", query_s.query.args[1], Term('[]'))
        nn_query_2 = (MNIST_NETWORK_NAME, arg2)

        res = self.model.evaluate_nn([nn_query_1, nn_query_2])
        print(res)
        # Contains output distributions for first and second digit
        dist1: torch.Tensor = res[nn_query_1]
        dist2: torch.Tensor = res[nn_query_2]

        # Contains target
        target = query.query.args[query.output_ind[0]]
        oh_target = one_hot(torch.tensor(int(target)), self.max_classes)

        # Detach tensors to ensure importance sampler doesnt train it
        cat_in = torch.cat([dist1.detach(), dist2.detach(), oh_target])
        self.embedding = storch.Tensor(self.lin1(cat_in), [], [])
        print(self.embedding)

        # TODO:
        #  One-hot encode target (min 0, max 18)
        #  Pass into network together with output distrs for 1st anad 2nd digit for embedding
        #  Sample value for first digit with linear layer
        #  Use result together with embedding to sample 2nd digit
        #  Pass to inference engine


    def get_sample(self, term: Term) -> StochasticTensor:
        if self.sample1 is None:
            distr1 = OneHotCategorical(logits=self.outp1(self.embedding))
            method = self.method_factory(str(term), self.n)
            self.sample1 = method(distr1)
            return self.sample1
        else:
            distr2 = OneHotCategorical(logits=self.outp2(storch.cat([self.embedding, self.sample1], dim=-1)))

            method = self.method_factory(str(term), 1)
            sample2 = method(distr2)
            return sample2

    def update_sampler(self, found_results: torch.Tensor):
        pass