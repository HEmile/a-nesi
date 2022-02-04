from collections import OrderedDict
from typing import Optional, Dict, List

import torch

from deepproblog.query import Query

from deepproblog.sampling.sampler import QueryMapper
from problog.logic import Term


class Memoizer:
    def __init__(self, mapper: QueryMapper, size=10000):
        self.mapper = mapper
        self.memory: OrderedDict[str, bool] = OrderedDict()
        self.size = size

    def _to_string(self, q_i: List[Term], q_o: List[str], sample_map: OrderedDict[str, torch.Tensor], index: int):
        # We are not using the name of the input terms, because those differ over iterations (eg train(0) and train(2)...)
        id = "in{"
        for i, in_term in enumerate(sample_map.values()):
           id += f"{i}={torch.argmax(in_term[index]).numpy()};"
        id += "}out{"
        for i, out_term in enumerate(q_o):
            id += f"{i}={out_term};"
        id += "}"
        return id

    def lookup(self, query: Query, sample_map: OrderedDict[str, torch.Tensor]) -> List[Optional[int]]:
        from deepproblog.sampling.sample import COST_FOUND_PROOF, COST_NO_PROOF
        res = []
        n = next(sample_map.values().__iter__()).size(0)
        q_i, q_o = self.mapper(query)
        for i in range(n):
            as_string = self._to_string(q_i, q_o, sample_map, i)
            if as_string in self.memory:
                if self.memory[as_string]:
                    res.append(COST_FOUND_PROOF)
                else:
                    res.append(COST_NO_PROOF)
            else:
                res.append(None)
        return res

    def add(self, query: Query, sample_map: OrderedDict[str, torch.Tensor], costs: List[int]):
        from deepproblog.sampling.sample import COST_FOUND_PROOF
        q_i, q_o = self.mapper(query)
        for i in range(len(costs)):
            as_string = self._to_string(q_i, q_o, sample_map, i)
            if as_string in self.memory:
                # Needed to reset position in queue
                self.memory.move_to_end(as_string, last=True)
            else:
                self.memory[as_string] = costs[i] == COST_FOUND_PROOF

    def update(self):
        diff = len(self.memory) - self.size
        if diff > 0:
            for i in range(diff):
                self.memory.popitem(last=False)
