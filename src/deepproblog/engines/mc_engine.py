from typing import Optional

from deepproblog.engines.exact_engine import ExactEngine
from deepproblog.model import Model
from deepproblog.sampling.grad_estim import factory_storch_method
from deepproblog.sampling.sampler import Sampler, IndependentSampler

EXTERN = "{}_extern_nocache_"



class MCEngine(ExactEngine):
    def __init__(self, model: Model, sampler: Optional[Sampler]=None):
        super().__init__(model)
        self.sampler = sampler
        if not sampler:
            self.sampler = IndependentSampler(model, factory_storch_method(), 3)


    def use_circuits(self) -> bool:
        return False
