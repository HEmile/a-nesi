from typing import TYPE_CHECKING, Callable, Sequence, Dict, List, Tuple, OrderedDict

import itertools
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
QueryMapper = Callable[[Query], Tuple[List[Term], List[Term]]]


class DefaultQueryMapper(QueryMapper):
    def __call__(self, query: Query) -> Tuple[List[Term], List[Term]]:
        query_s = query.substitute()
        inputs = list(map(self._deconstruct_term, query_s.query.args))
        # Delete outputs
        for ind in query.output_ind:
            del inputs[ind]
        inputs = list(itertools.chain(*inputs))
        outputs = list(map(lambda ind: query.query.args[ind], query.output_ind))
        return inputs, outputs

    def _deconstruct_term(self, term: Term) -> List[Term]:
        if term.functor == '.':
            return list(itertools.chain(*map(self._deconstruct_term, term.args)))
        if term.functor == '[]':
            return []

        return [Term(".", term, Term('[]'))]

def print_samples(samples: List):
    for od in samples:
        samplez = []
        for sample in od.values():
            samplez.append(torch.argmax(sample, dim=-1))
        print(samplez)


class Sampler(torch.nn.Module):
    model: "Model"
    network_name: str
    mapper: QueryMapper

    def __init__(self, method_factory: "MethodFactory", n: int, n_classes_query:int, entropy_weight: float, mapper: QueryMapper=None):
        super().__init__()
        if not mapper:
            self.mapper = DefaultQueryMapper()
        self.method_factory: "MethodFactory" = method_factory
        self.n = n
        self.n_classes_query = n_classes_query
        self.entropy_weight = entropy_weight

    def sample_atoms(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
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
            if q_o[0].is_constant():
                target_l.append(one_hot(torch.tensor(int(q_o[0])), self.n_classes_query))
            else:
                # If no supervision signal is given, just predict every option
                target_l.append(torch.ones(self.n_classes_query))

        n_i_terms = len(inputs[0])

        res = self.model.evaluate_nn(to_eval)
        dists: [storch.Tensor] = []
        self.samples = []
        for index_term in range(n_i_terms):
            d_tensors = [res[to_eval[n_i_terms * i + index_term]] for i in range(len(queries))]
            dist = torch.stack(d_tensors)
            # Assume first dimension (the one stacked over) is batch dimension. Is this correct?
            dist = storch.denote_independent(dist, 0, 'batch')
            dists.append(dist)
        method: Method = self.create_method(dists, target_l)
        for index_term, dist in enumerate(dists):
            # print(dist)
            # Call storchastic sample method
            # TODO: How to combine this with MAPO? We already implemented the memoizer, can we move that to storch?
            # def print_grad(x):
            #     print(torch.sum(x, dim=0))
            # dist.register_hook(print_grad)

            # TEST: Try to maximize entropy
            entropy = -(dist * (dist + 1e-9).log()).sum(-1)
            # print(storch.reduce_plates(entropy))
            # 0.1 worked fine for score function
            # storch.add_cost(-self.entropy_weight*entropy, f'entropy_{index_term}')

            sample = self._sample(dist, index_term, method)
            self.samples.append(sample)
            sample_t = sample._tensor
            if not sample.plates[0].name == 'batch':
                sample_t = sample_t.permute((1, 0, 2))
            for i, sample_dict in enumerate(samples):
                sample_dict[str(Term(*to_eval[n_i_terms * i + index_term]))] = sample_t[i]
        # print_samples(samples)

    def _sample(self, dist: torch.Tensor, index_term: int, method: Method):
        return method(OneHotCategorical(dist))

    def create_method(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Method:
        return self.method_factory("z", self.n)

    def __call__(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
        self.sample_atoms(queries, samples)

    def update_sampler(self, found_results: torch.Tensor):
        pass

    def parents(self) -> List[storch.Tensor]:
        return self.samples

    def get_hyperparameters(self) -> dict:
        return {
            "n": self.n,
            "type": "independent"
        }

