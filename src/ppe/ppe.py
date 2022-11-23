from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Optional

import torch
from torch import nn

from ppe.nrm import NRMBase, ST, O, W
from ppe.fit_dirichlet import fit_dirichlet
from torch.distributions import Categorical

class PPEBase(ABC, Generic[ST]):

    def __init__(self, nrm: NRMBase[ST], perception: nn.Module, amount_samples: int, belief_size:List[int],
                 initial_concentration: float = 0.1, K_beliefs: int = 100):
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

        # We're training these two models separately, so let's also use two different optimizers.
        #  This ensures we won't accidentally update the wrong model.
        self.nrm_optimizer = torch.optim.Adam(self.nrm.parameters())
        self.perception_optimizer = torch.optim.Adam(self.perception.parameters())

        self.alpha = torch.ones((1, len(belief_size), max(belief_size))) * initial_concentration
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
        p_P = fit_dirichlet(beliefs, self.alpha)
        P = p_P.sample((self.amount_samples,))
        p_w = Categorical(probs=P)
        w = p_w.sample()

        y = self.symbolic_function(w)

        log_p = p_w.log_prob(w)

        # TODO: Compute the log-probability of w and y using the NRM
        #  Should probably create a state with both w and y as constraints
        #  Or handle constraints differently!
        initial_state = self.initial_state(P, y, w)
        log_q = self.nrm(initial_state)
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
        if not self.beliefs:
            self.beliefs = P
        else:
            self.beliefs = torch.cat((self.beliefs, P), dim=0)
            if self.beliefs.shape[0] > self.K_beliefs:
                self.beliefs = self.beliefs[-self.K_beliefs:]

        self.nrm_optimizer.zero_grad()
        nrm_loss = self.nrm_loss(self.beliefs)

        # TODO: We gotta be sure this only changes the NRM parameters
        nrm_loss.backward()

        self.nrm_optimizer.step()

        self.perception_optimizer.zero_grad()
        initial_state = self.initial_state(P, y, generate_w=False)
        log_q_y = self.nrm(initial_state)
        log_q_y.backward()
        self.perception_optimizer.step()

        return nrm_loss, log_q_y

    @abstractmethod
    def initial_state(self, P: torch.Tensor, y: Optional[torch.Tensor]=None, w: Optional[torch.Tensor]=None, generate_w=True) -> ST:
        assert not (y is None and w is not None)
        pass

    @abstractmethod
    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        pass
