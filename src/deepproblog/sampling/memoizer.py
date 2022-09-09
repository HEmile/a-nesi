from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, List, Set, TYPE_CHECKING
if TYPE_CHECKING:
    from sample import ProofResult

import torch

from deepproblog.query import Query

from deepproblog.sampling.sampler import QueryMapper
from problog.logic import Term

@dataclass
class Memoized:
    found_solution: bool
    solution: Optional[str]
    counterExamples: Optional[Set[str]] = None


class Memoizer:
    """
    Memorizes whether different samples (that is, prolog programs!) can prove the query.
    Sampled prolog programs + their query are represented as a string to allow efficient lookup.
    The main use for this is reusing previous proofs: This process is deterministic, so we don't need to recompute it.

    Additionally, we optionally save prolog programs that prove the query for the MAPO algorithm.
    TODO: Assumes there is only one query that is correct
    """
    def __init__(self, mapper: QueryMapper, memoize_proofs=True, size=10000):
        self.mapper = mapper
        self.memory: OrderedDict[str, Memoized] = OrderedDict()
        self.size = size
        self.proof_memoizer: Dict[str, Set[str]] = {}
        self.memoize_proofs = memoize_proofs


    def _to_string_in(self, sample_map: OrderedDict[str, torch.Tensor], index: int):
        id = ''
        # TODO: Strings can be shorter by removing the index= here, since it doesn't add anything.
        #  This just iterates... Can use that to reduce memory overhead
        for i, in_term in enumerate(sample_map.values()):
            if i > 0:
                id += ';'
            # TODO: I kind of forgot what this argmax did
            id += f"{i}={torch.argmax(in_term[index]).numpy()}"
        return id

    def from_string(self, id) -> List[int]:
        return list(map(lambda s: int(s.split('=')[1]), id.split(';')))

    def _to_string_out(self, q_o: List[Term]) -> str:
        id = ''
        for i, out_term in enumerate(q_o):
            if i > 0:
                id += ';'
            id += f"{i}={out_term}"
        return id

    def query_to_string(self, query: Query) -> str:
        _, q_o = self.mapper(query)
        return self._to_string_out(q_o)

    def lookup(self, sample_map: OrderedDict[str, torch.Tensor]) -> List[Optional[Memoized]]:
        res = []
        n = next(sample_map.values().__iter__()).size(0)
        for i in range(n):
            as_string_in = self._to_string_in(sample_map, i)
            if as_string_in in self.memory:
                res.append(self.memory[as_string_in])
            else:
                res.append(None)
        return res

    def add(self, query: Query, sample_map: OrderedDict[str, torch.Tensor], results: ProofResult):
        from deepproblog.sampling.sample import COST_FOUND_PROOF

        _, q_o = self.mapper(query)

        n = len(results.costs) if results.costs else len(results.completions)
        for i in range(n):
            proof_string = self._to_string_in(sample_map, i)
            memo = self.memory.get(proof_string, None)

            if memo:
                # Needed to reset position in queue
                self.memory.move_to_end(proof_string, last=True)

            if results.costs:
                query_string = self._to_string_out(q_o)
                found_proof = results.costs[i] == COST_FOUND_PROOF
                if memo and not memo.found_solution:
                    if found_proof:
                        memo.counterExamples = None
                        memo.found_solution = True
                        memo.solution = query_string
                    else:
                        memo.counterExamples.add(query_string)
                elif not memo:
                    if found_proof:
                        self.memory[proof_string] = Memoized(True, solution=query_string)
                        if self.memoize_proofs:
                            if query_string in self.proof_memoizer:
                                self.proof_memoizer[query_string].add(proof_string)
                            else:
                                self.proof_memoizer[query_string] = {proof_string}
                    else:
                        self.memory[proof_string] = Memoized(False, None, counterExamples={query_string})
            elif not memo:
                # Not a ground query, but we have not saved the proof yet
                query_string = self._to_string_out(self.mapper(Query(results.completions[i]))[1])
                self.memory[proof_string] = Memoized(True, solution=query_string)


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
            return list(map(self.from_string, self.proof_memoizer[query_string]))
        return []

