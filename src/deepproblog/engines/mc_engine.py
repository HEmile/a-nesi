from typing import Optional

from deepproblog.engines.exact_engine import ExactEngine
from deepproblog.model import Model
from deepproblog.sampling.grad_estim import factory_storch_method
from deepproblog.sampling.sampler import Sampler

EXTERN = "{}_extern_nocache_"



class MCEngine(ExactEngine):
    def __init__(self, model: Model):
        super().__init__(model)
        for network in model.networks.values():
            if not network.sampler:
                network.sampler = Sampler(factory_storch_method(), 3)
            network.sampler.model = model

    def use_circuits(self) -> bool:
        return False