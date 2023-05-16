import storch
from anesi import ANeSIBase
import torch.nn as nn
import torch


class RLOOWrapper(nn.Module):

    def __init__(self, anesi: ANeSIBase, amount_samples: int, lr: float):
        super().__init__()
        self.anesi = anesi
        self.amount_samples = amount_samples
        self.method = storch.method.ScoreFunction("w", n_samples=amount_samples, baseline_factory="batch_average")
        self.optimizer = torch.optim.Adam(self.anesi.perception.parameters(), lr=lr)

    def structure_y(self, y: torch.Tensor) -> torch.Tensor:
        y = storch.deterministic(self.anesi.preprocess_y)(y)
        y = torch.stack(y, dim=-1)
        return y

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        y = storch.denote_independent(y, 0, "b")
        P = self.anesi.perception(x)
        P = storch.denote_independent(P, 0, "b")
        w = self.method(torch.distributions.Categorical(probs=P))

        y_pred = storch.deterministic(self.anesi.symbolic_function)(w)
        # y_pred = self.structure_y(y_pred)
        # y = self.structure_y(y)

        hamming_loss: storch.Tensor = torch.ne(y_pred, y).float().mean(-1)
        storch.add_cost(hamming_loss, "loss")
        norm_loss = storch.backward()
        self.optimizer.step()
        return norm_loss
