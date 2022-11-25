from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Optional
import torch
from torch import nn
from ppe.nrm import NRMBase, ST, O, W
from ppe.fit_dirichlet import fit_dirichlet
from torch.distributions import Categorical


class PPEBase(ABC, Generic[ST]):

    def __init__(self, nrm: NRMBase[ST], perception: nn.Module, amount_samples: int, belief_size:List[int],
                 initial_concentration: float = 500, dirichlet_iters: int = 50, dirichlet_lr: float = 1.0, K_beliefs: int = 100,
                 nrm_lr= 1e-3, perception_lr= 1e-3):
        """
        :param nrm: The neurosymbolic reverse model
        :param perception: The perception network. Should accept samples from data
        :param amount_samples: The amount of samples to draw to train the NRM
        :param initial_concentration: The initial concentration of the Dirichlet distribution
        :param K_beliefs: The amount of beliefs to keep to fit the Dirichlet
        """
        self.nrm = nrm
        self.perception = perception
        self.amount_samples = amount_samples
        self.belief_size = belief_size
        self.K_beliefs = K_beliefs
        self.dirichlet_iters = dirichlet_iters
        self.dirichlet_lr = dirichlet_lr

        # We're training these two models separately, so let's also use two different optimizers.
        #  This ensures we won't accidentally update the wrong model.
        self.nrm_optimizer = torch.optim.Adam(self.nrm.parameters(), lr=nrm_lr)
        self.perception_optimizer = torch.optim.Adam(self.perception.parameters(), lr=perception_lr)

        self.alpha = torch.ones((len(belief_size), max(belief_size))) * initial_concentration
        self.alpha.requires_grad = True
        self.beliefs = None

    def sample(self, x: torch.Tensor) -> ST:
        """
        Algorithm 1
        Sample from the PPE model
        :param data: The data to sample from
        :return: A sample from the PPE model
        """
        P = self.perception(x)

        initial_state = self.initial_state(P)

        return self.nrm(initial_state, amount_samples=self.amount_samples)

    def nrm_loss(self, beliefs: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 2
        For now assumes all w_i are the same size
        """
        p_P = fit_dirichlet(beliefs, self.alpha, self.dirichlet_lr, self.dirichlet_iters)
        P = p_P.sample((self.amount_samples,))
        p_w = Categorical(probs=P)
        # (batch, |W|)
        w = p_w.sample()

        # (batch,)
        y = self.symbolic_function(w)

        # (batch,)
        log_p = p_w.log_prob(w).sum(-1)

        initial_state = self.initial_state(P, y, w)
        result = self.nrm.forward(initial_state)
        log_q = torch.stack(result.forward_probabilities, -1).log().sum(-1)
        return (log_q - log_p).pow(2).mean()

    def train(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 3
        - Calls algorithm 2 (nrm_loss)
        - Minimizesnrm  loss using first optimizer
        - Computes q(y|P)
        - Maximize log prob
        """
        P = self.perception(x)
        if self.beliefs is None:
            self.beliefs = P
        else:
            self.beliefs = torch.cat((self.beliefs, P), dim=0)
            if self.beliefs.shape[0] > self.K_beliefs:
                self.beliefs = self.beliefs[-self.K_beliefs:]

        self.nrm_optimizer.zero_grad()
        nrm_loss = self.nrm_loss(self.beliefs)

        # TODO: We gotta be sure this only changes the NRM parameters
        # TODO: Figure out how to retain the graph. Maybe it's not needed since P is not involved.
        nrm_loss.backward()

        self.nrm_optimizer.step()
        self.nrm_optimizer.zero_grad()

        self.perception_optimizer.zero_grad()
        initial_state = self.initial_state(P, y, generate_w=False)
        result = self.nrm.forward(initial_state)
        stack_ys = torch.stack(result.forward_probabilities, -1).log()
        log_q_y = stack_ys.sum(-1).mean()
        loss_percept = -log_q_y
        loss_percept.backward()
        self.perception_optimizer.step()

        return nrm_loss, loss_percept, P

    @abstractmethod
    def initial_state(self, P: torch.Tensor, y: Optional[torch.Tensor]=None, w: Optional[torch.Tensor]=None, generate_w=True) -> ST:
        assert not (y is None and w is not None)
        pass

    @abstractmethod
    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        pass
