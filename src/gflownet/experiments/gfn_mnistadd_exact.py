# an exact GFlowNet sampler for mnist Add
from typing import List, Tuple

import torch
from torch.distributions import Categorical


class GFlowNetExact:
    def __init__(self, N):
        self.N = N

    def sample(self, p1, p2, query, mode='positives', amt_samples=1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.N == 1:
            # For every option, count the number of models that would be consistent with it
            model_c = self.count_models([], query)
            if mode == 'positives':
                # Adjust the probabilities to be consistent with the model counts
                unnorm_p = p1 * model_c
                p = unnorm_p / unnorm_p.sum(-1).unsqueeze(-1)
                # Sample from the adjusted probabilities
                d1 = Categorical(p).sample((amt_samples,))
                # Compute the other number, which is unique given the first
                d2 = query - d1
                assert (d1 + d2 == query).all()
                return [d1], [d2]
            if mode == 'negatives':
                # There are 10 worlds for every option. For some, there is a valid world. We want to filter those out.
                unnorm_p = p1 * (10 - model_c)
                p = unnorm_p / unnorm_p.sum()
                d1 = Categorical(p).sample()
                # There can be a world that is valid. We should filter it out
                pos = query - d1
                p2 = torch.tensor(p2)
                for i in range(pos.shape[0]):
                    if 0 <= pos[i] < 10:
                        p2[i, pos[i]] = 0

                # Renormalize probabilities after filtering
                p2 = p2 / p2.sum(-1).unsqueeze(-1)
                d2 = Categorical(p2).sample()
                assert (d1 + d2 != query).any()
                return [d1], [d2]

        raise NotImplementedError()


    def count_models(self, sampled_digits: List[int], query: torch.Tensor) -> torch.Tensor:
        if self.N == 1:
            # We'll do other cases later
            assert not sampled_digits or len(sampled_digits) == 0
            query = query.unsqueeze(-1)
            # return query - 9 <= \
            rang = torch.arange(10, device=query.device).unsqueeze(0)
            first_comp = rang <= query
            second_comp = rang >= query - 9
            return torch.logical_and(first_comp, second_comp).int()
        raise NotImplementedError()