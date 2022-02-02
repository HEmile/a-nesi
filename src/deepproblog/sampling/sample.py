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
from itertools import product

import time
import torch
from problog.clausedb import ClauseDB

from problog.logic import Term
from problog.program import LogicProgram

from typing import Sequence, TYPE_CHECKING, Dict, List

import storch
from deepproblog.engines.builtins import embed
from deepproblog.query import Query
from deepproblog.sampling.parallel import run_proof_threads, _single_proof, ProofProcessData
from deepproblog.semiring import Result

if TYPE_CHECKING:
    from deepproblog.model import Model

def run_proofs_sync(program: LogicProgram, sample_map: List[Dict[Term, torch.Tensor]], batch: Sequence[Query], n: int) -> List[List[int]]:
    return list(map(_single_proof,
                          [ProofProcessData(program, sample_map[i], batch[i], j)
                           for i, j in product(range(len(batch)), range(n))]))


# noinspection PyUnusedLocal
def estimate(model: "Model", program: ClauseDB, batch: Sequence[Query],
             propagate_evidence=False, amount_workers=4, **kwdargs) -> List[Result]:
    # Initial version will not support evidence propagation.
    from collections import defaultdict

    results = []

    # TODO: Calculate how long this takes
    start_time = time.time()
    sample_map: List[Dict[str, torch.Tensor]] = [OrderedDict() for _ in range(len(batch))]
    # TODO: What if there are multiple networks + samplers?
    for network in model.networks.values():
        sampler = network.sampler
        sampler.sample(batch, sample_map)
        break
    all_costs = [[1 for j in range(sampler.n)] for i in range(len(batch))]
    # This map gets reused over multiple samples of the same query, so we do not query the NN model unnecessarily


    # with ThreadPoolExecutor() as executor:
    #     executor.map(lambda indexes:
    #                  _single_proof(program, sampler, sample_map, batch[indexes[0]], indexes[1], indexes[0],
    #                                all_costs),
    #                  product(range(len(batch)), range(sampler.n)))

    # TODO: No idea if this is a good idea but w/e
    #  Needed because we dont want to pickle the neural networks
    _networks = model.networks
    model.networks = {}

    all_costs = run_proof_threads(program, sample_map, batch, sampler.n)
    # all_costs = run_proofs_sync(program, sample_map, batch, sampler.n)

    # print(all_costs)
    model.networks = _networks

    # print(all_costs)
    if sampler.is_batched():
        query_time = time.time() - start_time
        parents: [storch.StochasticTensor] = []
        for network in model.networks.values():
            parents.extend(network.sampler.parents())

        cost_tensor = torch.tensor(all_costs)
        plates = parents[-1].plates
        batch_first = plates[0].name == 'batch'
        if not batch_first:
            cost_tensor = cost_tensor.T
        sampler.update_sampler(cost_tensor)
        # TODO: May not work if the sampling method has more than 2 dimensions
        found_proof = storch.Tensor(cost_tensor, parents, plates, "found_proof")
        results.append(
            Result({}, found_proof=found_proof, ground_time=query_time, stoch_tensors=parents, is_batched=True))

    return results
