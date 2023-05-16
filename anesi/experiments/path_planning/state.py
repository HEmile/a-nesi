from typing import List, Optional

import torch
from torch import Tensor

from inference_models import Constraint, StateBase

EPS = 1E-6


class SPState(StateBase):

    def __init__(self,
                 probability: torch.Tensor,
                 constraint: Constraint,
                 N: int,
                 y: List[Tensor] = [],
                 w: List[Tensor] = [],
                 generate_w: bool = True,
                 final: bool = False):
        # Assuming probability is a b x 2*N x 10 Tensor
        # state: Contains the sampled digits
        self.pw = probability

        self.constraint = constraint
        self.y = y
        self.w = w
        self.N = N

        super().__init__(generate_w, final)

    def log_p_world(self) -> torch.Tensor:
        world = self.w
        world_tensor = torch.cat(world, dim=-1).unsqueeze(-1)
        log_pw = (self.pw + EPS).log()
        # sum = 0.
        # for i, d in enumerate(world):
        #     sum += (self.pw[:, i] + EPS).log().gather(1, d)
        gathered_tensor = torch.gather(log_pw, -1, world_tensor)
        return gathered_tensor.squeeze().sum(-1)

    def finished_generating_y(self) -> bool:
        return len(self.y) == 1

    def next_state(self, action: torch.Tensor, beam_selector: Optional[torch.Tensor] = None) -> StateBase:
        if not self.finished_generating_y():
            return SPState(self.pw, self.constraint, self.N, [action], self.w, self.generate_w, False)

        w = self.w
        if beam_selector is not None:
            w = list(map(lambda wi: wi.gather(-1, beam_selector), w))
        return SPState(self.pw, self.constraint, self.N, self.y, w + [action], self.generate_w, True)

    def symbolic_pruner(self) -> torch.Tensor:
        raise NotImplementedError()


directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1], [0, 0]]
EOS = len(directions) - 1


def transport_y(direction, prev_y: torch.Tensor, N: int, assert_in_bounds=True):
    # Assume prev_y is a [b] tensor
    coord_prev_y = torch.stack([prev_y // N, prev_y % N], dim=-1)

    dt = torch.tensor(directions, device=prev_y.device)
    coord_next_y = coord_prev_y + dt[direction]
    next_y = coord_next_y[..., 0] * N + coord_next_y[..., 1]
    if assert_in_bounds:
        assert (coord_next_y >= 0).all() and (coord_next_y < N).all()
    else:
        next_y[coord_next_y[..., 1] < 0] = -1
        next_y[coord_next_y[..., 1] >= N] = -1
        next_y[coord_next_y[..., 0] < 0] = -1
        next_y[coord_next_y[..., 0] >= N] = -1
    return next_y


def all_transport_y(prev_y: torch.Tensor, N: int):
    all_dirs = torch.arange(0, len(directions) - 1, device=prev_y.device).expand(prev_y.shape[0], -1)
    transported = transport_y(all_dirs, prev_y.unsqueeze(1), N, assert_in_bounds=False)
    return transported


class SPStatePath(SPState):

    def __init__(self,
                 probability: torch.Tensor,
                 constraint: Constraint,
                 N: int,
                 y: List[Tensor] = [],
                 w: List[Tensor] = [],
                 generate_w: bool = True,
                 final: bool = False,
                 prev_ys: Optional[Tensor] = None):
        # Here, y is a list of actions that indicate the directions. -1 is no action after the goal is reached (ie padding)
        # prev_last_y is a [b] tensor that indicates the last position of the agent
        # prev_grid_y is a [b, N*N] tensor that indicates the path so far
        super().__init__(probability, constraint, N, y, w, generate_w, final)

        self.parallel_y = constraint[0] and len(constraint[0]) > 0
        if prev_ys is None:
            self.prev_ys = torch.zeros((probability.shape[0], 1), device=probability.device, dtype=torch.long)
            self.constraint_coords = None
            if self.parallel_y:
                constraint_dirs = constraint[0][0]
                amt_directions = constraint_dirs.shape[1]
                self.constraint_coords = torch.zeros((probability.shape[0], amt_directions,), device=probability.device, dtype=torch.long)
                coord = self.constraint_coords[:, 0]

                for i in range(amt_directions - 1):
                    coord = transport_y(constraint_dirs[..., i], coord, N)
                    self.constraint_coords[:, i + 1] = coord

        elif not self._finished_y(prev_ys[-1]) and not self.parallel_y:
            new_y = transport_y(self.y[-1], prev_ys[..., -1], N)
            self.prev_ys = torch.cat([prev_ys, new_y.unsqueeze(-1)], dim=-1)
        else:
            self.prev_ys = prev_ys

    def _finished_y(self, last_y):
        return (last_y == self.N * self.N - 1).all() or self.parallel_y and len(self.y) == 1

    def finished_generating_y(self) -> bool:
        return self._finished_y(self.prev_ys[..., -1])

    def next_state(self, action: torch.Tensor, beam_selector: Optional[torch.Tensor] = None) -> StateBase:
        y = self.y
        if beam_selector is not None:
            y = list(map(lambda yi: yi.gather(-1, beam_selector), y))
        if not self.finished_generating_y():
            return SPStatePath(self.pw, self.constraint, self.N, y + [action], self.w, self.generate_w,
                               prev_ys=self.prev_ys)

        w = self.w
        if beam_selector is not None:
            w = list(map(lambda wi: wi.gather(-1, beam_selector), w))
        return SPStatePath(self.pw, self.constraint, self.N, self.y, w + [action], self.generate_w, True,
                           prev_ys=self.prev_ys)

    def _symbolic_pruner(self, prev_ys: torch.Tensor):
        last_y = prev_ys[..., -1]
        pruner = torch.zeros((self.pw.shape[0], len(directions)), device=self.pw.device)
        mask = last_y < self.N * self.N - 1
        # If we are at the goal, ensure the only valid action is to stay
        pruner[~mask, -1] = 1.
        transported = all_transport_y(prev_ys[..., -1][mask], self.N)
        # Ensure it is a valid action and we haven't been there before
        pruner[mask, :-1] = ((transported != -1) *
                             ~torch.any(transported.unsqueeze(-1) == prev_ys[mask].unsqueeze(-2), dim=-1)).float()
        return pruner

    def symbolic_pruner(self) -> torch.Tensor:
        if self.finished_generating_y():
            raise NotImplementedError()
        if self.parallel_y:
            collected_pruners = []
            for i in range(self.constraint_coords.shape[1]):
                collected_pruners.append(self._symbolic_pruner(self.constraint_coords[:, :i + 1]))
            return torch.stack(collected_pruners, dim=-2)
        else:
            return self._symbolic_pruner(self.prev_ys)
