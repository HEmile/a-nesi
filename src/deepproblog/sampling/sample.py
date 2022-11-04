#! /usr/bin/env python
"""
Query-based sampler for ProbLog 2.1+
===================================

  Concept:
      Each term has a value which is stored in the SampledFormula.
      When an probabilistic atom or choice atom is evaluated:
          - if probability is boolean discrete:
                          determine Yes/No, store in formula and return 0 (True) or None (False)
          - if probability is non-boolean discrete:
                          determine value, store value in formula and return key to value
      Adds builtin sample(X,S) that calls X and returns the sampled value in S.


  How to support evidence?
      => evidence on facts or derived from deterministic data: trivial, just fix value
      => evidence on atoms derived from other probabilistic facts:
          -> query evidence first
          -> if evidence is false, restart immediately
  Currently, evidence is supported through post-processing.


Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from collections import OrderedDict, defaultdict

import time
from dataclasses import dataclass

import torch

from deepproblog.engines.prolog_engine.swi_program import SWIProgram
from deepproblog.sampling.memoizer import Memoizer
from problog.clausedb import ClauseDB
from problog.logic import Term

from problog.program import LogicProgram

from typing import Sequence, TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import storch
from deepproblog.query import Query

from deepproblog.semiring import Result, Results

if TYPE_CHECKING:
    from deepproblog.model import Model

COST_FOUND_PROOF = 1
COST_NO_PROOF = -1


@dataclass
class ProofResult:

    mode: Literal['estimate', 'learn']

    completions: Optional[List[str]]

    costs: Optional[List[int]]

def _single_proof(program: LogicProgram, query: Query, sample_map: OrderedDict[str, torch.Tensor],
                  memoizer: Memoizer, n: int, mode: Literal['estimate', 'learn']) -> ProofResult:
    memoization = memoizer.lookup(sample_map)
    correct_out = memoizer.query_to_string(query)

    proof_results = ProofResult(mode, [] if mode == 'estimate' else None, None)

    is_ground_query = all(map(lambda t: t.is_constant(), memoizer.mapper(query)[1]))
    assert is_ground_query and mode == 'learn' or mode == 'estimate'

    if is_ground_query:
        proof_results.costs = [
            None if not memo else
            ((COST_FOUND_PROOF if memo.solution == correct_out else COST_NO_PROOF) if memo.found_solution else (
                COST_NO_PROOF if correct_out in memo.counterExamples else None
            )) for memo in memoization
        ]

        # If every result is memoized
        if len(list(filter(lambda c: c is None, proof_results.costs))) == 0:
            return proof_results
    else:
        # TODO: These memoizations (memo.solution) are of the form 0:index. These need to parse to a term together with the query
        proof_results.completions = [
            None if not memo else (
                (memo.solution if memo.found_solution else None)
            ) for memo in memoization
        ]

        # If every result is memoized
        if len(list(filter(lambda c: c is None, proof_results.completions))) == 0:
            return proof_results

    queryS = query.substitute().query

    for sample_count in range(n):
        # Check if memoized
        if is_ground_query and proof_results.costs[sample_count] is not None:
            continue
        if not is_ground_query and proof_results.completions[sample_count] is not None:
            continue
        # TODO: If not batched, ensure it gets result correctly

        # # TODO: Right now, the code assumes result.queries() only returns a single tuple. What if there is more?
        # for name, truth_value in result.queries():
        #     if is_ground_query:
        #         if name == queryS and truth_value == result.TRUE:
        #             proof_results.costs[sample_count] = COST_FOUND_PROOF
        #         else:
        #             proof_results.costs[sample_count] = COST_NO_PROOF
        #     else:
        #         assert truth_value == result.TRUE
        #         proof_results.completions[sample_count] = name

    memoizer.add(query, sample_map, proof_results)
    return proof_results

def run_proofs_sync(program: LogicProgram, sample_map: List[OrderedDict[str, torch.Tensor]],
                    batch: Sequence[Query], memoizer: Memoizer, n: int, mode: Literal["estimate", "learn"]) -> List[ProofResult]:
    return list(map(lambda i: _single_proof(program,  batch[i], sample_map[i], memoizer, n, mode), range(len(batch))))


# noinspection PyUnusedLocal
def dpl_mc(model: "Model", program: SWIProgram, batch: Sequence[Query], memoizer: Memoizer, mode: Literal["estimate", "learn"],
           propagate_evidence=False, amount_workers=4, parallel=False, **kwdargs) -> List[Result]:
    # Initial version will not support evidence propagation.
    start_time = time.time()

    # TODO: Calculate how long this takes
    # This map gets reused over multiple samples of the same query, so we do not query the NN model unnecessarily
    sample_map: List[OrderedDict[Term, torch.Tensor]] = [OrderedDict() for _ in range(len(batch))]

    # TODO: What if there are multiple networks + samplers?
    for network in model.networks.values():
        sampler = network.sampler
        sampler.sample_atoms(batch, sample_map)
        break

    n = next(iter(sample_map[0].values())).shape[0]
    # all_proofs = run_proofs_sync(program, sample_map, batch, memoizer, n, mode)

    parents: [storch.StochasticTensor] = []
    for network in model.networks.values():
        parents.extend(network.sampler.parents())

    found_proof = None
    plates = parents[-1].plates
    batch_first = plates[0].name == 'batch'

    results = []
    prob_tensor = None

    return None
    # if all_proofs[0].costs:
    #     # Convert the 'costs' (high if found no proof, otherwise low) into a Storch Tensor for averaging
    #     cost_tensor = torch.tensor(list(map(lambda r: r.costs, all_proofs)))
    #     if not batch_first:
    #         cost_tensor = cost_tensor.T
    #     sampler.update_sampler(cost_tensor)
    #     memoizer.update()
    #     # TODO: May not work if the sampling method has more than 2 dimensions
    #     found_proof = storch.Tensor(cost_tensor, parents, plates, "found_proof")
    # if all_proofs[0].completions:
    #     # Use Storchastic to average the different completions together. There might be duplicates!
    #     # TODO: Ensure that the hashing function properly recognizes duplicates.
    #     completion_to_index: Dict[Term, Tuple[int, int]] = dict()
    #     index_to_completion: List[Dict[int, Term]] = [dict() for _ in range(len(batch))]
    #     indexes = torch.zeros((len(batch), n), dtype=torch.long)
    #
    #     for batch_i, proof in enumerate(all_proofs):
    #         for j, completion in enumerate(proof.completions):
    #             if isinstance(completion, Term):
    #                 term = completion
    #             else:
    #                 # Transform the completion string into a term
    #                 query = batch[batch_i]
    #                 query_s = query.substitute().query
    #                 completion_ints = memoizer.from_string(completion)
    #                 substitution: Dict[str, Constant] = dict()
    #                 for output_ind in query.output_ind:
    #                     substitution[query_s.args[output_ind]] = Constant(completion_ints[output_ind])
    #                 term = query_s.apply(substitution)
    #
    #             # Assign the term into the right datastructure
    #             index = -1
    #             if completion in completion_to_index:
    #                 _, index = completion_to_index[term]
    #             else:
    #                 index = len(index_to_completion[batch_i])
    #                 index_to_completion[batch_i][index] = term
    #                 completion_to_index[term] = batch_i, index
    #             indexes[batch_i, j] = index
    #
    #     # Transform the different completions into a one-hot tensor
    #     if not batch_first:
    #         indexes = indexes.T
    #     one_hot = F.one_hot(indexes)
    #     one_hot = storch.Tensor(one_hot, parents, plates, "one_hot")
    #
    #     # Average the one-hot tensor to get a probability matrix. Uses Storchastic to do proper weighting
    #     prob_tensor = storch.reduce_plates(one_hot, 'z')
    #     _probs = prob_tensor.detach_tensor()
    #
    #     # Assign the probabilities to the right query
    #     for batch_i in range(len(batch)):
    #         r: Dict[Term, torch.Tensor] = {}
    #         if all_proofs[0].completions:
    #             dist = _probs[batch_i]
    #             index_to_completion_b = index_to_completion[batch_i]
    #             for j in range(dist.shape[-1]):
    #                 if j in index_to_completion_b:
    #                     r[index_to_completion_b[j]] = dist[j]
    #         results.append(r)
    # if len(results) == 0:
    #     results.append({})
    # query_time = time.time() - start_time
    #
    # result_l = list(map(lambda r: Result(r), results))
    # return Results(result_l,
    #                found_proof=found_proof, batch_time=query_time,
    #                stoch_tensors=parents, prob_tensor=prob_tensor)
