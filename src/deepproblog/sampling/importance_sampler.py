from typing import Callable, List, Sequence, Dict

import torch
from torch import detach

from torch.distributions import OneHotCategorical, Distribution

from deepproblog.query import Query
from deepproblog.sampling.sampler import Sampler
from storch import Plate
import storch
from torch.nn.functional import one_hot

from storch.method import Method
from storch.sampling.importance_sampling import ImportanceSampleDecoder
from storch.sampling.seq import AncestralPlate


class ImportanceSampler(Sampler):
    is_dists: List[Distribution]

    def sample(self, queries: Sequence[Query], sample_map: List[Dict[str, torch.Tensor]]):
        self.is_dists = []
        super().sample(queries, sample_map)
        # print("dist1", dist1)
        # p1.logits.register_hook(lambda g: print("logits1 grad", g))
        # p2.logits.register_hook(lambda g: print("logits2 grad", g))
        # p1.probs.register_hook(lambda g: print("probs1 grad", g))
        # p2.probs.register_hook(lambda g: print("probs2 grad", g))



    def _sample(self, dist_t: torch.Tensor, index_term: int, method: Method) -> storch.Tensor:
        return super()._sample(dist_t, index_term, method)

    def create_method(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Method:
        func = self.create_is_distribution_cb(dists, target_l)
        sampler = self

        def wrapped(prev_samples: List[storch.Tensor]) -> Distribution:
            d = func(prev_samples)
            sampler.is_dists.append(d)
            return d

        return self.method_factory("z", self.n, ImportanceSampleDecoder("z", self.n, wrapped))

    def create_is_distribution_cb(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Callable[[List[storch.Tensor]], Distribution]:
        # Compute the proposal distributions
        # Implement this with your importance sampler.
        pass

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
        for_is = []
        for i, sample in enumerate(self.samples):
            for_is.append(storch.StochasticTensor(sample._tensor, for_is, self.transform_plates(sample.plates),
                                                  sample.name, sample.n, self.is_dists[i], True, sample.method))
        new_cost = storch.Tensor(self.is_learning_rate_weight*found_results, for_is, for_is[-1].plates)
        # print(torch.mean(found_results.double()))
        # print("s1", for_is_1._tensor)
        # print("s2", for_is_2._tensor)
        # print("costs", found_results)
        storch.add_cost(new_cost, "found_cost")