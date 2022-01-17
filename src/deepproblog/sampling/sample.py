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
from typing import Sequence, TYPE_CHECKING, Tuple

from problog.engine_stack import FixedContext
from problog.errors import process_error, GroundingError
from problog.formula import LogicFormula
from problog.logic import Term, Constant, ArithmeticError, term2list
from problog.program import PrologFile, LogicProgram
from problog.tasks.sample import init_db, RateCounter, init_engine, verify_evidence, sample_poisson, FunctionStore, \
    SampledFormula
from torch.distributions import Categorical

from deepproblog.query import Query

if TYPE_CHECKING:
    from deepproblog.model import Model



class SampledFormulaDPL(SampledFormula):
    model: "Model"

    def __init__(self, model: "Model", **kwargs):
        SampledFormula.__init__(self, **kwargs)
        self.model = model
        self.sampled_nn_values = dict()


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
                    # TODO: Cache the results. If it is necessary, that is. (seems to be)
                    # TODO: This now assumes a categorical distribution. What about bernoulli?
                    to_evaluate = [(probability.args[0], probability.args[1])]
                    name = Term(probability.args[0], probability.args[1])
                    if name in self.sampled_nn_values:
                        # Already sampled
                        value = self.sampled_nn_values[name]
                        if value == probability.args[2]:
                            result_node = self.TRUE
                        else:
                            result_node = self.FALSE
                    else:
                        # Time to sample
                        res = self.model.evaluate_nn(to_evaluate)[to_evaluate[0]]
                        distr = Categorical(res)
                        sample = distr.sample()
                        prob = distr.log_prob(sample).exp()
                        sample = int(sample.detach().numpy())
                        prob = prob.detach().numpy()
                        print(identifier)
                        print(probability)
                        print(sample, prob)
                        self.probability *= prob
                        self.sampled_nn_values[name] = sample
                        if sample == probability.args[2]:
                            result_node = self.TRUE
                        else:
                            result_node = self.FALSE
                else:
                    value, prob = self.sample_value(probability)
                    self.probability *= prob
                    result_node = self.add_value(value)
                self.facts[identifier] = result_node
                print(identifier, result_node)
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
    #     for index, value in ev_target.lookup_evidence.items():
    #         node = ev_target.get_node(index)
    #         if ev_target.is_true(value):
    #             evidence_facts.append((node[0], 1.0) + node[2:])
    #         elif ev_target.is_false(value):
    #             evidence_facts.append((node[0], 0.0) + node[2:])
    # else:
    evidence_facts = []
    ev_target = None

    return db, evidence_facts, ev_target


def sample(
    model,
    n=1,
    format="str",
    propagate_evidence=False,
    distributions=None,
    progress=False,
    **kwdargs
):
    engine = init_engine(**kwdargs)
    db, evidence, ev_target = init_db(engine, model, propagate_evidence)
    i = 0
    r = 0

    if progress:
        rate = RateCounter()

    try:
        while i < n or n == 0:
            target = SampledFormula()
            if distributions is not None:
                target.distributions.update(distributions)

            for ev_fact in evidence:
                target.add_atom(*ev_fact)

            engine.functions = FunctionStore(target=target, database=db, engine=engine)
            # TODO!
            result = ground(engine, db, target=target)
            if verify_evidence(engine, db, ev_target, target):
                if format == "str":
                    yield result.to_string(db, **kwdargs)
                else:
                    yield result.to_dict()
                i += 1
            else:
                r += 1
            engine.previous_result = result
            if progress:
                rate.update()
    except KeyboardInterrupt:
        pass
    if r:
        logging.getLogger("problog_sample").info("Rejected samples: %s" % r)


# noinspection PyUnusedLocal
def estimate(model: "Model", program: LogicProgram, batch: Sequence[Query], n=0, propagate_evidence=False, **kwdargs):
    # Initial version will not support evidence propagation.
    from collections import defaultdict

    for query in batch:
        engine = init_engine(**kwdargs)
        db, evidence, ev_target = init_db(engine, program, propagate_evidence)

        start_time = time.time()
        estimates = defaultdict(float)
        counts = 0.0
        r = 0
        try:
            while n == 0 or counts < n:
                target = SampledFormulaDPL(model)
                # for ev_fact in evidence:
                #     target.add_atom(*ev_fact)

                queryS = query.substitute().query

                result: SampledFormulaDPL = ground(engine, db, query.substitute().query, target=target)

                # Evidence propagation & Evidence checking. To be implemented later.
                # if verify_evidence(engine, db, query):
                # else:
                #     r += 1

                if queryS in result.facts and result.facts[queryS]:
                    estimates[queryS] += 1.0
                counts += 1.0

                engine.previous_result = result
                print(result)
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass

        total_time = time.time() - start_time
        rate = counts / total_time
        print(
            "%% Probability estimate after %d samples (%.4f samples/second):"
            % (counts, rate)
        )
        print(estimates)

        if r:
            logging.getLogger("problog_sample").info("Rejected samples: %s" % r)

        for k in estimates:
            estimates[k] = estimates[k] / counts
    return estimates




