from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
from torch import nn

class GFlowNetBase(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, p1: torch.Tensor, p2: torch.Tensor, query: torch.Tensor, amt_samples=1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pass

    @abstractmethod
    def loss(self, success: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, *args):
        return self.sample(*args)