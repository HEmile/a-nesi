import torch
from problog.logic import Term

from deepproblog.examples.MNIST.network import MNIST_NETWORK_NAME
from deepproblog.query import Query
from deepproblog.sampling.sampler import Sampler
from storch import StochasticTensor
from storch.method import Method


class AdditionSampler(Sampler):
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
        print(res[nn_query_1], res[nn_query_2])

        # Contains target
        print(query.query.args[query.output_ind[0]])
        # TODO:
        #  One-hot encode target (min 0, max 18)
        #  Pass into network together with output distrs for 1st anad 2nd digit for embedding
        #  Sample value for first digit with linear layer
        #  Use result together with embedding to sample 2nd digit
        #  Pass to inference engine

        pass

    def get_sample(self, term: Term, method: Method) -> StochasticTensor:
        pass

    def update_sampler(self, found_results: torch.Tensor):
        pass