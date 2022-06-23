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
import logging
import random
from collections import OrderedDict, defaultdict

import time
import torch

from deepproblog.sampling.memoizer import Memoizer
from problog.clausedb import ClauseDB
from problog.engine_stack import FixedContext
from problog.errors import GroundingError
from problog.logic import Term

from problog.program import LogicProgram

from typing import Sequence, TYPE_CHECKING, Dict, List

import storch
from deepproblog.query import Query

from deepproblog.semiring import Result
from problog.tasks.sample import SampledFormula, init_engine

if TYPE_CHECKING:
    from deepproblog.model import Model

COST_FOUND_PROOF = 1
COST_NO_PROOF = -1


class SampledFormulaDPL(SampledFormula):
    model: "Model"

    def __init__(self, sample_map: Dict[str, torch.Tensor],
                 query_counts: int, **kwargs):
        SampledFormula.__init__(self, **kwargs)
        self.sample_map: Dict[str, torch.Tensor] = sample_map
        self.query_counts = query_counts

    def _is_nn_probability(self, probability: Term):
        return probability.functor == "nn"

    def add_atom(
            self,
            identifier: Term,
            probability: Term,
            group: (int, FixedContext) = None,
            name: Term = None,
            source=None,
            cr_extra=True,
            is_extra=False,
    ):
        # def add_atom(self, identifier, probability, group=None, name=None, source=None):
        if probability is None:
            return 0

        if group is None:  # Simple fact
            if identifier not in self.facts:
                if self._is_simple_probability(probability):
                    p = random.random()
                    prob = float(probability)
                    value = p < prob
                    if value:
                        result_node = self.TRUE
                        self.probability *= prob
                    else:
                        result_node = self.FALSE
                        self.probability *= 1 - prob
                else:
                    value, prob = self.sample_value(probability)
                    self.probability *= prob
                    result_node = self.add_value(value)
                self.facts[identifier] = result_node
                return result_node
            else:
                return self.facts[identifier]
        else:
            # choice = identifier[-1]
            origin = identifier[:-1]
            if identifier not in self.facts:
                if self._is_simple_probability(probability):
                    p = float(probability)
                    if origin in self.groups:
                        r = self.groups[origin]  # remaining probability in the group
                    else:
                        r = 1.0

                    if r is None or r < 1e-8:
                        # r is too small or another choice was made for this origin
                        value = False
                    else:
                        value = random.random() <= p / r
                    if value:
                        self.probability *= p
                        self.groups[
                            origin
                        ] = None  # Other choices in group are not allowed
                    elif r is not None:
                        self.groups[origin] = r - p  # Adjust remaining probability
                    if value:
                        result_node = self.TRUE
                    else:
                        result_node = self.FALSE
                elif self._is_nn_probability(probability):
                    # Sample from nn code.
                    # TODO: This now assumes a categorical distribution. What about bernoulli?
                    name = str(Term(probability.args[0], probability.args[1]))
                    if name not in self.sample_map:
                        raise AttributeError(f"NN atom {name} not in sample_map {self.sample_map}. Make sure to add it in your sampler.")
                    sample = self.sample_map[name]

                    # Lookup sample in sampled storch.Tensor
                    # if self.sampler.n == 1:
                    #     detach_sample = sample[self.batch_counts]
                    # TODO: What if query_counts is 0?
                    detach_sample = sample[self.query_counts]
                    # Disabled for performance reasons
                    # distr = sample.distribution
                    # prob = distr.log_prob(detach_sample).exp()
                    # if isinstance(prob, storch.Tensor):
                    #     prob = prob._tensor
                    # prob = prob.detach().numpy()
                    # self.probability *= prob
                    # TODO: Make sure this comparison is valid.
                    if detach_sample[int(probability.args[2])] == 1:
                        result_node = self.TRUE
                    else:
                        result_node = self.FALSE
                else:
                    value, prob = self.sample_value(probability)
                    self.probability *= prob
                    result_node = self.add_value(value)
                self.facts[identifier] = result_node
                return result_node
            else:
                return self.facts[identifier]


def ground(engine, db, query: Term, target, assume_prepared=True) -> SampledFormulaDPL:
    db = engine.prepare(db)

    # # Old loading queries code:
    # # Load queries: use argument if available, otherwise load from database.
    # queries = [q[0] for q in engine.query(db, Term("query", None))]

    # Evidence code
    # evidence = engine.query(db, Term("evidence", None, None))
    # evidence += engine.query(db, Term("evidence", None))
    # for ev in evidence:
    #     if not isinstance(ev[0], Term):
    #         raise GroundingError("Invalid evidence")  # TODO can we add a location?

    if not isinstance(query, Term):
        raise GroundingError("Invalid query")  # TODO can we add a location?

    # Ground queries
    # queries = [(target.LABEL_QUERY, q) for q in queries]
    logger = logging.getLogger("problog")
    logger.debug("Grounding query '%s'", query)
    target = engine.ground(db, query, target, label=target.LABEL_QUERY, assume_prepared=assume_prepared)
    logger.debug("Ground program size: %s", len(target))
    return target


def init_db(engine, model: LogicProgram, propagate_evidence=False):
    db = engine.prepare(model)

    # if propagate_evidence:
    #     evidence = engine.query(db, Term("evidence", None, None))
    #     evidence += engine.query(db, Term("evidence", None))
    #
    #     ev_target = LogicFormula()
    #     engine.ground_evidence(db, ev_target, evidence)
    #     ev_target.lookup_evidence = {}
    #     ev_nodes = [
    #         node
    #         for name, node in ev_target.evidence()
    #         if node != 0 and node is not None
    #     ]
    #     ev_target.propagate(ev_nodes, ev_target.lookup_evidence)
    #
    #     evidence_facts = []
    #     for query_counts, value in ev_target.lookup_evidence.items():
    #         node = ev_target.get_node(query_counts)
    #         if ev_target.is_true(value):
    #             evidence_facts.append((node[0], 1.0) + node[2:])
    #         elif ev_target.is_false(value):
    #             evidence_facts.append((node[0], 0.0) + node[2:])
    # else:
    evidence_facts = []
    ev_target = None

    return db, evidence_facts, ev_target

def _single_proof(program: LogicProgram, query: Query, sample_map: OrderedDict[str, torch.Tensor],
                  memoizer: Memoizer, n: int) -> List[int]:
    costs = memoizer.lookup(query, sample_map)

    # If every result is memoized
    if len(list(filter(lambda c: c is None, costs))) == 0:
        return costs

    engine = init_engine()

    # TODO: Does propagate evidence speed things up?
    db, evidence, ev_target = init_db(engine, program, False)  # Assume propagate evidence is false

    queryS = query.substitute().query

    for sample_count in range(n):
        # Check if memoized
        if costs[sample_count] is not None:
            continue
        # TODO: If not batched, ensure it gets result correctly
        target = SampledFormulaDPL(sample_map, sample_count)
        # for ev_fact in evidence:
        #     target.add_atom(*ev_fact)
        result: SampledFormulaDPL = ground(engine, db, queryS, target=target)

        for name, truth_value in result.queries():
            # Note: We are minimizing, so we have negative numbers for good results!
            if name == queryS and truth_value == result.TRUE:
                # costs[batch_index][sample_count] = -1
                costs[sample_count] = COST_FOUND_PROOF
            else:
                costs[sample_count] = COST_NO_PROOF
        engine.previous_result = result
    memoizer.add(query, sample_map, costs)
    return costs

def run_proofs_sync(program: LogicProgram, sample_map: List[OrderedDict[str, torch.Tensor]],
                    batch: Sequence[Query], memoizer: Memoizer, n: int) -> List[List[int]]:
    return list(map(lambda i: _single_proof(program,  batch[i], sample_map[i], memoizer, n), range(len(batch))))


# noinspection PyUnusedLocal
def estimate(model: "Model", program: ClauseDB, batch: Sequence[Query], memoizer: Memoizer,
             propagate_evidence=False, amount_workers=4, parallel=False, **kwdargs) -> List[Result]:
    # Initial version will not support evidence propagation.
    start_time = time.time()
    results = []

    # TODO: Calculate how long this takes
    sample_map: List[OrderedDict[str, torch.Tensor]] = [OrderedDict() for _ in range(len(batch))]

    # TODO: What if there are multiple networks + samplers?
    for network in model.networks.values():
        sampler = network.sampler
        sampler.sample(batch, sample_map)
        break
    all_costs = [[1 for j in range(sampler.n)] for i in range(len(batch))]
    # This map gets reused over multiple samples of the same query, so we do not query the NN model unnecessarily

    if parallel:
        from deepproblog.sampling.parallel import run_proof_threads
        all_costs = run_proof_threads(model, sample_map, batch, sampler.n)
    else:
        all_costs = run_proofs_sync(program, sample_map, batch, memoizer, sampler.n)

    parents: [storch.StochasticTensor] = []
    for network in model.networks.values():
        parents.extend(network.sampler.parents())

    cost_tensor = torch.tensor(all_costs)
    plates = parents[-1].plates
    batch_first = plates[0].name == 'batch'
    if not batch_first:
        cost_tensor = cost_tensor.T
    sampler.update_sampler(cost_tensor)
    memoizer.update()
    # TODO: May not work if the sampling method has more than 2 dimensions
    found_proof = storch.Tensor(cost_tensor, parents, plates, "found_proof")
    query_time = time.time() - start_time
    results.append(
        Result({}, found_proof=found_proof, ground_time=query_time, stoch_tensors=parents, is_batched=True))


    return results
