from itertools import product

import logging

import random

import torch
from pathos.multiprocessing import ProcessPool
from problog.logic import Term
from typing import Dict, List, Sequence

from problog.engine_stack import FixedContext
from problog.errors import GroundingError
from problog.program import LogicProgram
from problog.tasks.sample import init_db, init_engine, SampledFormula

from deepproblog.query import Query


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

class ProofProcessData:
    def __init__(self, program, sample_map: Dict[str, torch.Tensor], query: Query, n: int, **kwargs):
        super().__init__(**kwargs)
        self.program = program
        self.sample_map = sample_map
        self.query = query
        self.n = n


def _single_proof(param: ProofProcessData) -> List[int]:#self, program, sampler: Sampler, sample_map: Dict, query: Query, sample_count: int, batch_index: int):
    # print("Entering!")
    engine = init_engine()
    # TODO: Does propagate evidence speed  things up?
    db, evidence, ev_target = init_db(engine, param.program, False)  # Assume propagate evidence is false

    queryS = param.query.substitute().query

    costs = []
    for sample_count in range(param.n):
        # TODO: If not batched, ensure it gets result correctly
        target = SampledFormulaDPL(param.sample_map, sample_count)
        # for ev_fact in evidence:
        #     target.add_atom(*ev_fact)
        result: SampledFormulaDPL = ground(engine, db, queryS, target=target)

        for name, truth_value in result.queries():
            # Note: We are minimizing, so we have negative numbers for good results!
            if name == queryS and truth_value == result.TRUE:
                # costs[batch_index][sample_count] = -1
                costs.append(-1)
            else:
                costs.append(0)
        engine.previous_result = result
    # print("Exiting!")
    return costs

def run_proof_threads(program: LogicProgram, sample_map: List[Dict[str, torch.Tensor]], batch: Sequence[Query], n: int) -> List[List[int]]:

    with ProcessPool() as p:
        all_costs = p.map(_single_proof,
                          [ProofProcessData(program, sample_map[i], batch[i], n)
                           for i in range(len(batch))])
    return all_costs