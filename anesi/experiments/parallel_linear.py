import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init
import torch.nn.functional as F


class ParallelLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, heads: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.weight = Parameter(torch.empty((heads, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(heads, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, parallel=True, index: int=-1) -> Tensor:
        if not parallel:
            assert index > -1
        else:
            assert index == -1
        if index > -1:
            assert not parallel

        if parallel:
            # TODO: Might be a good idea to test this code somehow
            a = input.unsqueeze(-2)
            b = self.weight.transpose(1, 2).unsqueeze(0)
            matm = torch.matmul(a, b).squeeze(-2)
            return matm + self.bias
        return F.linear(input, self.weight[index], self.bias[index])

    def extra_repr(self) -> str:
        return 'heads={}, in_features={}, out_features={}, bias={}'.format(
            self.heads, self.in_features, self.out_features, self.bias is not None
        )

def _calculate_fan_in_and_fan_out(tensor):
    num_input_fmaps = tensor.size(2)
    num_output_fmaps = tensor.size(1)
    receptive_field_size = 1

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
