from problog.logic import Clause, Term
from problog.program import SimpleProgram, LogicProgram

from deepproblog.engines.exact_engine import ExactEngine

EXTERN = "{}_extern_nocache_"



class MCEngine(ExactEngine):
    def use_circuits(self) -> bool:
        return False

    # def prepare(self, db) -> LogicProgram:
    #     translated = SimpleProgram()
    #     for e in db:
    #         new_es = [e]
    #         if type(e) is Term or type(e) is Clause:
    #             p = e.probability
    #             if p is not None and p.functor == "nn":
    #                 if len(p.args) == 4:
    #                     new_es = self.create_nn_predicate_ad(e)
    #                 elif len(p.args) == 3:
    #                     new_es = self.create_nn_predicate_det(e)
    #                 elif len(p.args) == 2:
    #                     new_es = self.create_nn_predicate_fact(e)
    #                 else:
    #                     raise ValueError(
    #                         "A neural predicate with {} arguments is not supported.".format(
    #                             len(p.args)
    #                         )
    #                     )
    #         for new_e in new_es:
    #             translated.add_clause(new_e)
    #     translated.add_clause(
    #         Clause(
    #             Term("_directive"),
    #             Term("use_module", Term("library", Term("lists.pl"))),
    #         )
    #     )
    #     return translated