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
import time
import torch
from problog.clausedb import ClauseDB
from problog.engine_stack import FixedContext
from problog.errors import GroundingError
from problog.logic import Term
from problog.program import LogicProgram
from problog.tasks.sample import init_db, init_engine, SampledFormula
from torch.distributions import Distribution, OneHotCategorical
from typing import Sequence, TYPE_CHECKING, Dict, List, Optional

import storch
from deepproblog.query import Query
from deepproblog.sampling.sampler import Sampler, DefaultSampler
from deepproblog.semiring import Result
from storch import StochasticTensor, CostTensor
from storch.method import Baseline
from storch.method.baseline import BatchAverageBaseline

if TYPE_CHECKING:
    from deepproblog.model import Model



class SampledFormulaDPL(SampledFormula):
    model: "Model"

    def __init__(self, model: "Model", sampler: Sampler, sample_map: Dict[Term, storch.StochasticTensor], method_factory, query_counts: int, **kwargs):
        SampledFormula.__init__(self, **kwargs)
        self.model = model
        self.sampler = sampler
        self.sample_map: Dict[Term, storch.StochasticTensor] = sample_map
        self.method_factory = method_factory
        self.query_counts = query_counts


    def _is_nn_probability(self, probability: Term):
        return probability.functor == "nn"

    def add_atom(
        self,
        identifier: Term,
        probability: Term,
        group: (int, FixedContext)=None,
        name: Term=None,
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
                    name = Term(probability.args[0], probability.args[1])
                    if name not in self.sample_map:
                        sample = self.sampler.get_sample(probability, self.method_factory("z"))
                        self.sample_map[name] = sample
                    sample = self.sample_map[name]
                    # Lookup sample in sampled storch.Tensor
                    distr = sample.distribution
                    if sample.n == 1:
                        detach_sample = sample._tensor
                    else:
                        detach_sample: torch.Tensor = sample._tensor[self.query_counts]
                    prob = distr.log_prob(detach_sample).exp()
                    prob = prob.detach().numpy()
                    self.probability *= prob
                    self.sample_map[name] = sample
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

class HybridBaseline(Baseline):
    def __init__(self):
        super().__init__()
        self.batch_avg_baseline = BatchAverageBaseline()
        # Const baseline of -0.1 since we want to punish whenever it samples all 0s
        self.const = torch.tensor(-0.1)

    def compute_baseline(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> torch.Tensor:
        if (cost_node._tensor < 0).any():
            return self.batch_avg_baseline.compute_baseline(tensor, cost_node)
        return self.const

def factory_storch_method(n: int, name="hybrid-baseline"):
    def create_storch_method(atom_name: str):
        if name == 'hybrid-baseline':
            return storch.method.ScoreFunction(atom_name, n_samples=n, baseline_factory=lambda s, c: HybridBaseline())
        return storch.method.ScoreFunction(atom_name, n_samples=n, baseline_factory='batch-average')
    return create_storch_method


# noinspection PyUnusedLocal
def estimate(model: "Model", program: ClauseDB, batch: Sequence[Query], n=0, propagate_evidence=False, sampler: Optional[Sampler]=None, **kwdargs) -> List[Result]:
    # Initial version will not support evidence propagation.
    from collections import defaultdict

    results = []

    if not sampler:
        sampler = DefaultSampler(model)

    for node in program.iter_nodes():
        if type(node).__name__ == 'choice':
            print(node)
    print("-----------------------")
    for query in batch:
        estimates = defaultdict(float)
        start_time = time.time()
        engine = init_engine(**kwdargs)
        db, evidence, ev_target = init_db(engine, program, propagate_evidence)

        queryS = query.substitute().query
        query_counts = 0
        # r = 0

        # This map gets reused over multiple samples of the same query, so we do not query the NN model unnecessarily
        sample_map = {}
        costs = []
        while n == 0 or query_counts < n:
            target = SampledFormulaDPL(model, sampler, sample_map, factory_storch_method(n), query_counts)
            # for ev_fact in evidence:
            #     target.add_atom(*ev_fact)
            result: SampledFormulaDPL = ground(engine, db, queryS, target=target)

            for name, truth_value in result.queries():
                # Note: We are minimizing, so we have negative numbers for good results!
                if name == queryS and truth_value == result.TRUE:
                    estimates[queryS] += 1.0
                    costs.append(-1)
                else:
                    costs.append(0)
            query_counts += 1

            engine.previous_result = result

        # if r:
        #     logging.getLogger("problog_sample").info("Rejected samples: %s" % r)

        assert len(sample_map) > 0
        cost_tensor = torch.tensor(costs)
        parents: [storch.Tensor] = list(sample_map.values())
        found_proof = storch.Tensor(cost_tensor, parents, parents[0].plates, "found_proof")

        estimates[queryS] = estimates[queryS] / query_counts
        found_proof = {queryS: found_proof}
        query_time = time.time() - start_time
        results.append(Result(estimates, found_proof=found_proof, ground_time=query_time, stoch_tensors=parents))

    return results




