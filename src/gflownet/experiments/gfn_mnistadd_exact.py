# an exact GFlowNet sampler for mnist Add
import torch

from gflownet import GFlowNetBase
from gflownet.experiments.state import MNISTAddState


class GFlowNetExact(GFlowNetBase):

    def flow(self, state: MNISTAddState) -> torch.Tensor:
        constraint = state.constraint
        ds = state.state
        if state.N == 1:
            # We'll do other cases later

            if constraint is None:
                # Should return the amount of models for each query
                raise NotImplementedError()

            if len(ds) == 0:
                query = constraint.unsqueeze(-1)
                rang = torch.arange(10, device=query.device).unsqueeze(0)
                first_comp = rang <= query
                second_comp = rang >= query - 9
                return torch.logical_and(first_comp, second_comp).int()
            else:
                d2 = constraint.unsqueeze(-1) - ds[0]
                # TODO: This assumes for every d1 there is a solution (ie 0 <= constraint - d1 <= 9)
                return torch.nn.functional.one_hot(d2, 10).int()

        raise NotImplementedError()

    def loss(self, final_state: MNISTAddState, is_wmc=False) -> torch.Tensor:
        # This is already perfect, no need to train it :)
        return torch.tensor(0.)
