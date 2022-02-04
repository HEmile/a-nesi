from typing import List, Callable

import torch
from torch import nn, detach

from deepproblog.sampling.grad_estim import MethodFactory
from deepproblog.sampling.importance_sampler import ImportanceSampler
from torch.distributions import OneHotCategorical, Distribution

import storch



class AdditionSampler(ImportanceSampler):
    def __init__(self, method: MethodFactory, n: int):
        super().__init__(method, n)
        self.max_classes = 19
        self.lin1 = nn.Linear(2 * 10 + self.max_classes, 30)
        self.outp1 = nn.Linear(30, 10)
        self.outp2 = nn.Linear(30 + 10, 10)
        self.is_learning_rate_weight = 10.

    def create_is_distribution_cb(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Callable[[List[storch.Tensor]], Distribution]:
        sampler = self
        outputs = storch.denote_independent(torch.stack(target_l), 0, 'batch')

        # Detach tensors to ensure importance sampler doesnt train it
        cat_in = storch.cat(list(map(detach, dists)) + [outputs], dim=-1)
        embedding = self.lin1(cat_in)

        def callback(prev_samples: List[storch.Tensor]):
            # TODO: This is not yet a general model
            if len(prev_samples) == 0:
                logits_1 = sampler.outp1(embedding)
                # logits_1.register_hook(lambda g: print("is1", g))
                dist1 = OneHotCategorical(logits=logits_1)
                # print(sampler.outp1.weight)
                return dist1
            else:
                logits_2: torch.Tensor = self.outp2(storch.cat([embedding, prev_samples[0]], dim=-1))
                # logits_2.register_hook(lambda g: print("s2", g))
                dist2 = OneHotCategorical(logits=logits_2)
                # print(sampler.outp2.weight)
                return dist2
        return callback

    def get_hyperparameters(self) -> dict:
        hps = super().get_hyperparameters()
        hps["type"] = "MNIST_importance"
        return hps
