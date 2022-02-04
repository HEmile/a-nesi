import multiprocessing

import time

import torch

from deepproblog.network import Network
from deepproblog.sampling.sample import _single_proof
from typing import Dict, List, Sequence

from problog.program import LogicProgram

from deepproblog.query import Query
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from problog import Model


class ProofProcessData:
    def __init__(self, program: LogicProgram, networks: List[Network], sample_map: Dict[str, torch.Tensor], query: Query, n: int, **kwargs):
        super().__init__(**kwargs)
        self.sample_map = sample_map
        self.networks = networks
        self.query = query
        self.n = n
        self.program = program
        self.load = True

def _single_proof_parallel(param: ProofProcessData) -> List[int]:
    # t = time.time()
    from deepproblog.model import Model
    from deepproblog.engines.mc_engine import MCEngine
    model = Model(param.program, param.networks, load=param.load)
    program = MCEngine(model).prepare(model.program)
    # print("prepare time", time.time() - t)

    return _single_proof(program, param.query, param.sample_map, param.n)


# def get_state_cdb(self):
#     return self.__dict__
#
# def set_state_cdb(self, d):
#     self.__dict__ = d
#
# ClauseDB.__setstate__ = get_state_cdb
# ClauseDB.__setstate__ = set_state_cdb


def run_proof_threads(model: "Model", sample_map: List[Dict[str, torch.Tensor]], batch: Sequence[Query], n: int) -> List[List[int]]:
    # I tried it out a lot but it seems to just be a lot slower. Might be better for harder problems
    # For MNIST 1 it is not worth it.

    t = time.time()
    networks = list(map(lambda n: Network(None, n.name, k=n.k), model.networks.values()))

    # Needed to prevent pickl errors
    for network in networks:
        network.function = None
    with multiprocessing.Pool() as p:
        all_costs = p.map(_single_proof_parallel,
                          [ProofProcessData("models/addition.pl", networks, sample_map[i], batch[i], n) for i in range(len(batch))])

    print("Total time", time.time() -t)
    return all_costs