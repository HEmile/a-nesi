from collections import OrderedDict
from typing import Optional, Dict, List, Set

import torch

from deepproblog.query import Query

from deepproblog.sampling.sampler import QueryMapper
from problog.logic import Term


class Memoizer:
    """
    Memorizes whether different samples (that is, prolog programs!) can prove the query.
    Sampled prolog programs + their query are represented as a string to allow efficient lookup.
    The main use for this is reusing previous proofs: This process is deterministic, so we don't need to recompute it.

    Additionally, we optionally save prolog programs that prove the query for the MAPO algorithm.
    """
    def __init__(self, mapper: QueryMapper, memoize_proofs=True, size=10000):
        self.mapper = mapper
        self.memory: OrderedDict[str, bool] = OrderedDict()
        self.size = size
        self.memoize_proofs = memoize_proofs
        self.proof_memoizer: Dict[str, Set[str]] = {}


    def _to_string_in(self, sample_map: OrderedDict[str, torch.Tensor], index: int):
        id = ''
        # TODO: Strings can be shorter by removing the index= here, since it doesn't add anything.
        #  This just iterates... Can use that to reduce memory overhead
        for i, in_term in enumerate(sample_map.values()):
            if i > 0:
                id += ';'
            id += f"{i}={torch.argmax(in_term[index]).numpy()}"
        return id

    def _from_string_in(self, id) -> List[int]:
        return list(map(lambda s: int(s.split('=')[1]), id.split(';')))

    def _to_string_out(self, q_o: List[str]):
        id = ''
        for i, out_term in enumerate(q_o):
            if i > 0:
                id += ';'
            id += f"{i}={out_term}"
        return id

    def _to_string(self, q_i: List[Term], q_o: List[str], sample_map: OrderedDict[str, torch.Tensor], index: int):
        # We are not using the name of the input terms, because those differ over iterations (eg train(0) and train(2)...)
        # End results are deterministic given an output.
        return f"I[{self._to_string_in(sample_map, index)}]O[{self._to_string_out(q_o)}]"

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
                found_proof = costs[i] == COST_FOUND_PROOF
                self.memory[as_string] = found_proof
                if found_proof and self.memoize_proofs:
                    query_string = self._to_string_out(q_o)
                    proof_string = self._to_string_in(sample_map, i)
                    if query_string in self.proof_memoizer:
                        self.proof_memoizer[query_string].add(proof_string)
                    else:
                        self.proof_memoizer[query_string] = {proof_string}

    def update(self):
        diff = len(self.memory) - self.size
        if diff > 0:
            for i in range(diff):
                self.memory.popitem(last=False)

    def get_proofs(self, query: Query) -> List[List[int]]:
        assert self.memoize_proofs
        q_i, q_o = self.mapper(query)
        query_string = self._to_string_out(q_o)
        if query_string in self.proof_memoizer:
            return list(map(self._from_string_in, self.proof_memoizer[query_string]))
        return []

