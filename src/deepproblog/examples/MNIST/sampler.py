from typing import Callable, List, Sequence, Dict

import torch
from problog.logic import Term
from torch.distributions import OneHotCategorical, Distribution

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

from storch.sampling.importance_sampling import ImportanceSampleDecoder


class AdditionSampler(Sampler):
    samples: Dict[str, storch.StochasticTensor]

    def __init__(self, model: Model, method: MethodFactory, n: int):
        super().__init__(model, method, n)
        self.max_classes = 19
        self.lin1 = nn.Linear(2 * 10 + self.max_classes, 30)
        self.outp1 = nn.Linear(30, 10)
        self.outp2 = nn.Linear(30 + 10, 10)


    def prepare_sampler(self, queries: Sequence[Query]):
        # TODO: Do this in batches of queries.
        #  When implementing that, pass it as a list into evaluate_nn, as it will automatically batch things.

        to_eval = []
        target_l = []
        for query in queries:
            query_s = query.substitute()
            arg1 = Term(".", query_s.query.args[0], Term('[]'))
            to_eval.append((MNIST_NETWORK_NAME, arg1))
            arg2 = Term(".", query_s.query.args[1], Term('[]'))
            to_eval.append((MNIST_NETWORK_NAME, arg2))

            # Contains target
            target = query.query.args[query.output_ind[0]]
            target_l.append(one_hot(torch.tensor(int(target)), self.max_classes))

        res = self.model.evaluate_nn(to_eval)
        d1_tensors = [res[to_eval[2*i]] for i in range(int(len(res) / 2))]
        d2_tensors = [res[to_eval[2*i + 1]] for i in range(int(len(res) / 2))]

        dist1 = storch.denote_independent(torch.stack(d1_tensors), 0, "batch")
        dist2 = storch.denote_independent(torch.stack(d2_tensors), 0, "batch")

        # Contains output distributions for first and second digit
        distr1 = OneHotCategorical(dist1)
        distr2 = OneHotCategorical(dist2)

        outputs = storch.denote_independent(torch.stack(target_l), 0, 'batch')

        # Detach tensors to ensure importance sampler doesnt train it
        cat_in = storch.cat([dist1.detach(), dist2.detach(), outputs], dim=-1)
        embedding = self.lin1(cat_in)

        sampler = self

        # Compute the proposal distributions
        def is_callback(prev_samples: List[storch.Tensor]) -> Distribution:
            if len(prev_samples) == 0:
                return OneHotCategorical(logits=sampler.outp1(embedding))
            else:
                return OneHotCategorical(logits=sampler.outp2(storch.cat([embedding, prev_samples[0]], dim=-1)))

        method = self.method_factory("z", self.n, ImportanceSampleDecoder("z", self.n, is_callback))

        sample1 = method(distr1)
        sample2 = method(distr2)

        self.samples = {str(Term(*to_eval[i])): sample1 if i % 2 == 0 else sample2 for i in range(int(len(res)))}

        # TODO:
        #  One-hot encode target (min 0, max 18)
        #  Pass into network together with output distrs for 1st anad 2nd digit for embedding
        #  Sample value for first digit with linear layer
        #  Use result together with embedding to sample 2nd digit
        #  Pass to inference engine

    # def _construct_method(self, term: Term, conditional: bool, proposal_distr: Distribution) -> Method:
    #     n = 1 if conditional else self.n
    #     plate_name = str(term)
    #     sampling_method = ImportanceSampling(plate_name, n, proposal_distr)
    #     return self.method_factory(plate_name, n, sampling_method)

    def get_sample(self, term: Term) -> StochasticTensor:
        name = Term(*term.args[0:2])
        return self.samples[str(name)]
        # if self.sample1 is None:
        #     self.sample1 = self.method(self.distr1)
        #     return self.sample1
        # else:
        #     sample2 = self.method(self.distr2)
        #     return sample2

    def update_sampler(self, found_results: torch.Tensor):
        pass
