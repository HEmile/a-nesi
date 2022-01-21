from typing import Optional

from problog.logic import Clause, Term
from problog.program import SimpleProgram, LogicProgram

from deepproblog.engines.exact_engine import ExactEngine
from deepproblog.model import Model
from deepproblog.sampling.sampler import Sampler

EXTERN = "{}_extern_nocache_"



class MCEngine(ExactEngine):
    def __init__(self, model: Model, sampler: Optional[Sampler]=None):
        super().__init__(model)
        self.sampler = sampler


    def use_circuits(self) -> bool:
        return False
