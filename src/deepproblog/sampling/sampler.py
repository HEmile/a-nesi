from typing import TYPE_CHECKING, Callable, Sequence, Dict, List, Tuple

import random

from distutils.dist import Distribution

import torch

from deepproblog.engines.builtins import one_hot
from problog.logic import Term
from torch.distributions import OneHotCategorical

from deepproblog.query import Query
from storch import StochasticTensor, storch
from storch.method import Method

if TYPE_CHECKING:
    from deepproblog.model import Model
    from deepproblog.sampling.sample import MethodFactory


# These map queries to input terms and outputs
QueryMapper = Callable[[Query], Tuple[List[Term], List[str]]]


class DefaultQueryMapper(QueryMapper):
    def __call__(self, query: Query):
        query_s = query.substitute()
        inputs = list(map(lambda arg: Term(".", arg, Term('[]')), query_s.query.args))
        # Delete outputs
        for ind in query.output_ind:
            del inputs[ind]
        outputs = list(map(lambda ind: query.query.args[ind], query.output_ind))
        return inputs, outputs


class Sampler(torch.nn.Module):
    model: "Model"
    network_name: str
    mapper: QueryMapper
    def __init__(self, method_factory: "MethodFactory", n: int, n_classes_query:int, mapper: QueryMapper=None):
        super().__init__()
        if not mapper:
            self.mapper = DefaultQueryMapper()
        self.method_factory: "MethodFactory" = method_factory
        self.n = n
        self.n_classes_query = n_classes_query

    def sample(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
        # TODO: For MAPO, we need to only sample from whatever is not memoized
        to_eval = []
        target_l: List[torch.Tensor] = []
        inputs: List[List[Term]] = []
        for query in queries:
            q_i, q_o = self.mapper(query)
            inputs.append(q_i)
            for term in q_i:
                to_eval.append((self.network_name, term))

            # TODO: How to generalize this?
            #  It assumes always only 1 target
            target_l.append(one_hot(torch.tensor(int(q_o[0])), self.n_classes_query))

        n_i_terms = len(inputs[0])

        res = self.model.evaluate_nn(to_eval)
        dists: [storch.Tensor] = []
        self.samples = []
        for index_term in range(n_i_terms):
            d_tensors = [res[to_eval[n_i_terms * i + index_term]] for i in range(int(len(res) / 2))]
            dist = torch.stack(d_tensors)
            # Assume first dimension (the one stacked over) is batch dimension. Is this correct?
            dist = storch.denote_independent(dist, 0, 'batch')
            dists.append(dist)
        method: Method = self.create_method(dists, target_l)
        for index_term, dist in enumerate(dists):
            # Call storchastic sample method
            # TODO: How to combine this with MAPO? We already implemented the memoizer, can we move that to storch?
            sample = self._sample(dist, index_term, method)
            self.samples.append(sample)
            sample_t = sample._tensor
            if not sample.plates[0].name == 'batch':
                sample_t = sample_t.permute((1, 0, 2))
            for i, sample_dict in enumerate(samples):
                sample_dict[str(Term(*to_eval[n_i_terms * i + index_term]))] = sample_t[i]

    def _sample(self, dist: torch.Tensor, index_term: int, method: Method):
        return method(OneHotCategorical(dist))

    def create_method(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Method:
        return self.method_factory("z", self.n)

    def __call__(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
        self.sample(queries, samples)

    def update_sampler(self, found_results: torch.Tensor):
        pass

    def parents(self) -> List[storch.Tensor]:
        return self.samples

    def get_hyperparameters(self) -> dict:
        return {
            "n": self.n,
            "type": "independent"
        }

