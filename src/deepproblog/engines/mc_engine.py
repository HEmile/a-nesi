from typing import Optional

from deepproblog.engines.exact_engine import ExactEngine
from deepproblog.model import Model
from deepproblog.sampling.grad_estim import factory_storch_method
from deepproblog.sampling.sampler import Sampler, IndependentSampler

EXTERN = "{}_extern_nocache_"



class MCEngine(ExactEngine):
    def __init__(self, model: Model, sampler: Optional[Sampler]=None):
        super().__init__(model)
        for network in model.networks.values():
            if not network.sampler:
                network.sampler = IndependentSampler(factory_storch_method(), 3)
            network.sampler.model = model

    def use_circuits(self) -> bool:
        return False