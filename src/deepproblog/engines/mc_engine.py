from problog.logic import Clause, Term
from problog.program import SimpleProgram, LogicProgram

from deepproblog.engines.exact_engine import ExactEngine

EXTERN = "{}_extern_nocache_"



class MCEngine(ExactEngine):
    def use_circuits(self) -> bool:
        return False
