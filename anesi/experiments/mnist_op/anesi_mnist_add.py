import math
from typing import Optional, List

import torch

from experiments.mnist_op import MNISTModel
from experiments.mnist_op.state_mnist import MNISTState

EPS = 1E-6


class MNISTAddState(MNISTState):

    def symbolic_pruner(self) -> torch.Tensor:
        device = self.pw.device
        if len(self.y) < self.N + 1:
            # Label. Rarely filters anything
            if len(self.y) == 0:
                return torch.ones((2,), device=device).unsqueeze(0)
            onez = torch.ones((10,), device=device)
            if len(self.y) == self.N:
                onez_without_nine = torch.ones_like(onez)
                onez_without_nine[-1] = 0
                for i in range(len(self.y[0].shape)):
                    onez_without_nine = onez_without_nine.unsqueeze(0)
                    onez = onez.unsqueeze(0)
                # TODO: Test if this is ever nonzero
                is_9s = 1
                if self.N - 1 > 0:
                    is_9s = torch.prod(torch.stack([self.oh_state[i + 1][..., 9] for i in range(self.N - 1)], -1), -1)
                is_19s = self.y[0] * is_9s
                is_19s = is_19s.unsqueeze(-1)
                return is_19s * onez_without_nine + (1 - is_19s) * onez
            return onez
        kth_digit = len(self.w)
        if kth_digit < self.N:
            # First number
            nw = 0
            sum_y = torch.stack([10 ** (kth_digit - i + 1) * self.y[i] for i in range(kth_digit + 2)], -1).sum(-1)
            rang = torch.arange(10, device=device).unsqueeze(0)
            ny = sum_y.unsqueeze(-1)
            if kth_digit > 0:
                sum_w = torch.stack([10 ** (kth_digit - i) * self.w[i] for i in range(kth_digit)], -1).sum(-1)
                nw = sum_w.unsqueeze(-1)
                if len(nw.shape) > len(ny.shape):
                    ny = ny.unsqueeze(-1)

            first_comp = nw + rang <= ny

            # Note: This part is pretty confusing lol. Need to really explain this well in the paper
            # On the last digit we generate, we can at most have 10**N-1 - 1 to add to it to make up y
            #  Before that, the next digits can together sum to above 10**kth_digit to allow one step further.
            #  There is one more edge case. Consider y = 109. Taking the first 0, we can only have 9 to add to it,
            #  meaning the second digit has to be maximum (99), and 9 + 99 = only 108!
            #  So if the digit of y coming after the one considered is 9, _and_ y is even, we have to remove an option
            subtr = 1

            if kth_digit != self.N - 1:
                # Computes: Are all trailing digits 9s?
                subtr = torch.stack([(self.y[i] == 9).int() for i in range(kth_digit + 2, self.N + 1)], -1) \
                    .prod(-1) \
                    .unsqueeze(-1)
                if len(subtr.shape) < len(ny.shape):
                    subtr = subtr.unsqueeze(-1)
            second_comp = nw + rang >= ny - (10 ** (kth_digit + 1) - subtr)
            keep = torch.logical_and(first_comp, second_comp).int()
            return keep

        # Second number. Deterministic
        ny = self.query_to_number()
        n1 = torch.stack([10 ** (self.N - i - 1) * self.w[i] for i in range(self.N)], -1).sum(-1)
        if len(n1.shape) > len(ny.shape):
            ny = ny.unsqueeze(-1)
        n2 = ny - n1
        d = (torch.div(n2, (10 ** (2 * self.N - kth_digit - 1)), rounding_mode="floor") % 10).long()
        # print(n1, ny, self.y, self.constraint, n2, d)
        # TODO: This assumes for every d1 there is a solution (ie 0 <= constraint - d1 <= 9)
        return torch.nn.functional.one_hot(d, 10).int()

    def len_y_encoding(self):
        # TODO: Needs to take into account the y-encoding method
        return self.N + 1


class MNISTAddModel(MNISTModel):

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> MNISTAddState:

        _initialstate = super().initial_state(P, y, w, generate_w)
        return MNISTAddState(_initialstate.pw, _initialstate.N, _initialstate.constraint, _initialstate.y_dims, generate_w=generate_w)

    def op(self, n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        return n1 + n2

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        if y_encoding == "base10":
            return [1] + [10] * N
        elif y_encoding == "base2":
            max_y = 10 ** (N+1) - 1
            return [1] * math.ceil(math.log2(max_y))
        raise NotImplementedError
