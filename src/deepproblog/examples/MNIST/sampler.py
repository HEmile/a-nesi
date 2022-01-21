import torch
from problog.logic import Term

from deepproblog.examples.MNIST.network import MNIST_NETWORK_NAME
from deepproblog.query import Query
from deepproblog.sampling.sampler import Sampler
from storch import StochasticTensor
from storch.method import Method


class AdditionSampler(Sampler):
    def prepare_sampler(self, query: Query):
        query_s = query.substitute()
        arg1 = query_s.query.args[0]
        arg2 = query_s.query.args[1]
        d1 = self.model.get_tensor(arg1)
        d2 = self.model.get_tensor(arg2)
        print(d1, d2)
        res1 = self.model.evaluate_nn([MNIST_NETWORK_NAME, arg1])
        res2 = self.model.evaluate_nn([MNIST_NETWORK_NAME, arg2])
        print(res1, res2)
        pass

    def get_sample(self, term: Term, method: Method) -> StochasticTensor:
        pass

    def update_sampler(self, found_results: torch.Tensor):
        pass