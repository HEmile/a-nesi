from typing import Callable, List, Sequence, Dict

import torch
from problog.logic import Term
from torch.distributions import OneHotCategorical, Distribution

from deepproblog.query import Query
from deepproblog.sampling.grad_estim import MethodFactory
from deepproblog.sampling.sampler import Sampler
from storch import StochasticTensor, Plate
import torch.nn as nn
import storch
from torch.nn.functional import one_hot

from storch.sampling.importance_sampling import ImportanceSampleDecoder
from storch.sampling.seq import AncestralPlate


class AdditionSampler(Sampler):
    samples: Dict[str, storch.StochasticTensor]
    sample1: storch.StochasticTensor
    sample2: storch.StochasticTensor
    dist1: Distribution
    dist2: Distribution

    def __init__(self, method: MethodFactory, n: int):
        super().__init__(method, n)
        self.max_classes = 19
        self.lin1 = nn.Linear(2 * 10 + self.max_classes, 30)
        self.outp1 = nn.Linear(30, 10)
        self.outp2 = nn.Linear(30 + 10, 10)
        self.is_learning_rate_weight = 10.


    def sample(self, queries: Sequence[Query], sample_map: List[Dict[str, torch.Tensor]]):
        # TODO: Do this in batches of queries.
        #  When implementing that, pass it as a list into evaluate_nn, as it will automatically batch things.

        to_eval = []
        target_l = []
        for query in queries:
            query_s = query.substitute()
            arg1 = Term(".", query_s.query.args[0], Term('[]'))
            to_eval.append((self.network_name, arg1))
            arg2 = Term(".", query_s.query.args[1], Term('[]'))
            to_eval.append((self.network_name, arg2))

            # Contains target
            target = query.query.args[query.output_ind[0]]
            target_l.append(one_hot(torch.tensor(int(target)), self.max_classes))

        res = self.model.evaluate_nn(to_eval)
        d1_tensors = [res[to_eval[2*i]] for i in range(int(len(res) / 2))]
        d2_tensors = [res[to_eval[2*i + 1]] for i in range(int(len(res) / 2))]

        dist1 = storch.denote_independent(torch.stack(d1_tensors), 0, "batch")
        dist2 = storch.denote_independent(torch.stack(d2_tensors), 0, "batch")

        # dist1._tensor.register_hook(lambda g: print("d1", g))
        # dist2._tensor.register_hook(lambda g: print("d2", g))

        p1 = OneHotCategorical(probs=dist1)
        p2 = OneHotCategorical(probs=dist2)

        # print("dist1", dist1)
        # p1.logits.register_hook(lambda g: print("logits1 grad", g))
        # p2.logits.register_hook(lambda g: print("logits2 grad", g))
        # p1.probs.register_hook(lambda g: print("probs1 grad", g))
        # p2.probs.register_hook(lambda g: print("probs2 grad", g))

        outputs = storch.denote_independent(torch.stack(target_l), 0, 'batch')

        # Detach tensors to ensure importance sampler doesnt train it
        cat_in = storch.cat([dist1.detach(), dist2.detach(), outputs], dim=-1)
        embedding = self.lin1(cat_in)

        sampler = self

        # Compute the proposal distributions
        def is_callback(prev_samples: List[storch.Tensor]) -> Distribution:
            if len(prev_samples) == 0:
                logits_1 = sampler.outp1(embedding)
                # logits_1.register_hook(lambda g: print("is1", g))
                sampler.dist1 = OneHotCategorical(logits=logits_1)
                # print(sampler.outp1.weight)
                return sampler.dist1
            else:
                logits_2: torch.Tensor = sampler.outp2(storch.cat([embedding, prev_samples[0]], dim=-1))
                # logits_2.register_hook(lambda g: print("s2", g))
                sampler.dist2 = OneHotCategorical(logits=logits_2)
                # print(sampler.outp2.weight)
                return sampler.dist2

        method = self.method_factory("z", self.n, ImportanceSampleDecoder("z", self.n, is_callback))

        # Contains output distributions for first and second digit
        self.sample1 = method(p1)
        self.sample2 = method(p2)

        self.samples = {str(Term(*to_eval[i])): self.sample1 if i % 2 == 0 else self.sample2 for i in range(int(len(res)))}
        sample1t = self.sample1._tensor
        if not self.sample1.plates[0].name == 'batch':
            sample1t = sample1t.permute((1, 0, 2))
        sample2t = self.sample2._tensor
        if not self.sample2.plates[0].name == "batch":
            sample2t = sample2t.permute((1, 0, 2))
        for i, sample_dict in enumerate(sample_map):
            sample_dict[str(Term(*to_eval[2*i]))] = sample1t[i]
            sample_dict[str(Term(*to_eval[2*i + 1]))] = sample2t[i]

    def transform_plates(self, plates: List[Plate]):
        plates = list(plates)
        for i, plate in enumerate(plates):
            if plate.name == 'z':
                plate: AncestralPlate
                # Ancestral plate with wrong weight.
                # TODO: Weight wont work for sampling without replacement.
                plates[i] = AncestralPlate('z', self.n, plate.parents, plate.variable_index,
                                           plate.parent_plate, plate.selected_samples, None, torch.tensor(1. / self.n))
        return plates

    def update_sampler(self, found_results: torch.Tensor):
        # Creates a fake SCG for the IS training
        for_is_1 = storch.StochasticTensor(self.sample1._tensor, [], self.transform_plates(self.sample1.plates),
                                           self.sample1.name, self.sample1.n, self.dist1, True, self.sample1.method)
        for_is_2 = storch.StochasticTensor(self.sample2._tensor, [for_is_1], self.transform_plates(self.sample2.plates), self.sample2.name,
                                           self.sample2.n, self.dist2, True, self.sample2.method)
        new_cost = storch.Tensor(self.is_learning_rate_weight*found_results, [for_is_1, for_is_2], for_is_2.plates)
        # print(torch.mean(found_results.double()))
        # print("s1", for_is_1._tensor)
        # print("s2", for_is_2._tensor)
        # print("costs", found_results)
        storch.add_cost(new_cost, "found_cost")

    def parents(self) -> List[storch.Tensor]:
        return [self.sample1, self.sample2]

    def get_hyperparameters(self) -> dict:
        hps = super().get_hyperparameters()
        hps["type"] = "MNIST_importance"
        return hps
