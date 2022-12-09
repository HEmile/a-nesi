from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Optional
import torch
from torch import nn
from torch.nn.functional import one_hot, softplus

from nrm import NRMBase, ST, NRMResult
from fit_dirichlet import fit_dirichlet
from torch.distributions import Categorical

EPS = 1e-8


class PPEBase(ABC, Generic[ST]):

    def __init__(self,
                 nrm: NRMBase[ST],
                 perception: nn.Module,
                 amount_samples: int,
                 belief_size: List[int],
                 initial_concentration: float = 500,
                 dirichlet_iters: int = 50,
                 dirichlet_lr: float = 1.0,
                 dirichlet_L2: float = 0.0,
                 K_beliefs: int = 100,
                 nrm_lr=1e-3,
                 nrm_loss="mse",
                 policy: str = "both",
                 perception_lr=1e-3,
                 perception_loss='sampled',
                 percept_loss_pref=1.0,
                 device='cpu',
                 ):
        """
        :param nrm: The neurosymbolic reverse model
        :param perception: The perception network. Should accept samples from data
        :param amount_samples: The amount of samples to draw to train the NRM
        :param initial_concentration: The initial concentration of the Dirichlet distribution
        :param K_beliefs: The amount of beliefs to keep to fit the Dirichlet
        :param percept_loss_pref: When using perception_loss='both', this will prefer log-q if > 1.0, otherwise sampled
        """
        assert perception_loss in ['sampled', 'log-q', 'both']
        assert nrm_loss in ['mse', 'bce']
        assert policy in ['both', 'off', 'on']
        self.nrm = nrm
        self.perception = perception
        self.amount_samples = amount_samples
        self.belief_size = belief_size
        self.K_beliefs = K_beliefs
        self.dirichlet_iters = dirichlet_iters
        self.dirichlet_lr = dirichlet_lr
        self.dirichlet_L2 = dirichlet_L2
        self.nrm_loss = nrm_loss
        self.policy = policy
        self.perception_loss = perception_loss
        self.percept_loss_pref = percept_loss_pref

        # We're training these two models separately, so let's also use two different optimizers.
        #  This ensures we won't accidentally update the wrong model.
        self.nrm_optimizer = torch.optim.Adam(self.nrm.parameters(), lr=nrm_lr)
        self.perception_optimizer = torch.optim.Adam(self.perception.parameters(), lr=perception_lr)

        self.alpha = torch.ones((len(belief_size), max(belief_size)), device=device) * initial_concentration
        self.alpha.requires_grad = True
        self.beliefs = None

    def off_policy_loss(self, p_P: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 2
        For now assumes all w_i are the same size
        """

        P = p_P.sample((self.amount_samples,))
        p_w = Categorical(probs=P)
        # (batch, |W|)
        w = p_w.sample()

        # (batch,)
        y = self.symbolic_function(w)

        # (batch,)
        log_p = p_w.log_prob(w)


        initial_state = self.initial_state(P, y, w)
        result = self.nrm.forward(initial_state)
        log_q = (torch.stack(result.forward_probabilities, -1) + EPS).log()
        if self.nrm_loss == 'mse':
            return (log_q.sum(-1) - log_p.sum(-1)).pow(2).mean()
        elif self.nrm_loss == 'bce':
            return nn.BCELoss()(log_q.sum(-1).exp(), log_p.sum(-1).exp())
        raise NotImplementedError()

    def log_q_loss(self, P: torch.Tensor, y: torch.Tensor):
        """
        Perception loss that maximizes the log probability of the label under the NRM model.
        """
        initial_state = self.initial_state(P, y, generate_w=False)
        result = self.nrm.forward(initial_state)
        stack_ys = (torch.stack(result.forward_probabilities, -1) + EPS).log()
        log_q_y = stack_ys.sum(-1).mean()
        return -log_q_y

    def sampled_loss(self, P: torch.Tensor, y: torch.Tensor, compute_perception_loss: bool, compute_nrm_loss: bool) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perception loss that maximizes the log probability of the label under the NRM model.
        """
        initial_state = self.initial_state(P.detach(), y, generate_w=True)
        result = self.nrm.forward(initial_state, amt_samples=self.amount_samples)

        percept_loss = nrm_loss = 0.

        w = torch.stack(result.final_state.w, -1)
        w = w.permute(1, 0, 2)

        # Take sum of log probs over all dimensions
        log_p_w = Categorical(probs=P).log_prob(w).sum(-1).T

        if compute_perception_loss:
            # # TODO: This might underflow
            # q_y = torch.stack(
            #     result.forward_probabilities[:len(result.final_state.y)], 1
            # ).prod(-1).detach()
            prediction = self.symbolic_function(torch.stack(result.final_state.w, -1))
            successes = prediction == y.unsqueeze(-1)
            # print(y)
            if self.nrm.prune:
                # If we prune, we know we are successful by definition
                assert successes.all()
                percept_loss = -log_p_w.mean()
            else:
                percept_loss = -(log_p_w * successes).mean() if successes.any() else 0.

        if compute_nrm_loss:
            log_q_y = (torch.stack(result.forward_probabilities[:len(result.final_state.y)], 1) + EPS).log().sum(-1)
            log_q_y = log_q_y.unsqueeze(-1)
            log_q_w_y = (torch.stack(result.forward_probabilities[len(result.final_state.y):], -1) + EPS).log().sum(-1)
            log_q = log_q_y + log_q_w_y
            log_p_w = log_p_w.detach()
            if self.nrm_loss == 'mse':
                nrm_loss = (log_q - log_p_w).pow(2).mean()
            elif self.nrm_loss == 'bce':
                nrm_loss = nn.BCELoss()(log_q.exp(), log_p_w.exp())
            else:
                raise NotImplementedError()

        return percept_loss, nrm_loss

    def train(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 3
        - Calls algorithm 2 (nrm_loss)
        - Minimizes nrm loss using first optimizer
        - Computes q(y|P)
        - Maximize log prob
        """
        P = self.perception(x)
        self.nrm_optimizer.zero_grad()
        nrm_loss = 0.
        use_off_policy_loss = not self.policy == 'on'
        if use_off_policy_loss or self.perception_loss == 'both':
            if self.beliefs is None:
                self.beliefs = P
            else:
                self.beliefs = torch.cat((self.beliefs, P), dim=0)
                if self.beliefs.shape[0] > self.K_beliefs:
                    self.beliefs = self.beliefs[-self.K_beliefs:]

            beliefs = self.beliefs.detach()
            p_P = fit_dirichlet(beliefs, self.alpha, self.dirichlet_lr, self.dirichlet_iters, self.dirichlet_L2)

            if use_off_policy_loss:
                _nrm_loss = self.off_policy_loss(p_P)
                _nrm_loss.backward()
                nrm_loss += _nrm_loss
                self.nrm_optimizer.step()
                self.nrm_optimizer.zero_grad()

        use_sampled_loss = not self.perception_loss == 'log-q'
        use_log_q_loss = not self.perception_loss == 'sampled'
        use_on_policy_loss = not self.policy == 'off'

        loss_sampled = 0.
        loss_log_q = 0.

        self.perception_optimizer.zero_grad()
        if use_sampled_loss or use_on_policy_loss:
            _loss_sampled, _nrm_loss = self.sampled_loss(P, y, use_sampled_loss, use_on_policy_loss)
            if use_sampled_loss:
                loss_sampled = _loss_sampled
            if use_on_policy_loss:
                _nrm_loss.backward()
                nrm_loss += _nrm_loss
                self.nrm_optimizer.step()
        if use_log_q_loss:
            loss_log_q = self.log_q_loss(P, y)

        w_sampled = 1. if use_sampled_loss else 0.
        w_log_q = 1. if use_log_q_loss else 0.
        # if self.perception_loss == 'both':
        #     w_log_q = torch.sigmoid(torch.log(self.percept_loss_pref * softplus(self.alpha.mean()))).detach()
        #     w_sampled = 1. - w_log_q
        loss_percept = w_sampled * loss_sampled + w_log_q * loss_log_q
        loss_percept.backward()
        self.perception_optimizer.step()

        return nrm_loss, loss_percept

    def test(self, x: torch.Tensor, y: torch.Tensor, true_w: Optional[List[torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Algorithm 1
        Sample from the PPE model
        :param x: The data to perform beam search on
        :param y: The true label y to predict and compare to
        :param true_w: The true w (explanation to get to y). Used to evaluate if explanations are correct
        """
        P = self.perception(x)
        initial_state = self.initial_state(P)
        result: NRMResult[ST] = self.nrm.beam(initial_state, beam_size=self.amount_samples)
        successes = self.success(result.final_state.y, y, beam=True).float()

        prior_predictions = torch.argmax(P, dim=-1)
        prior_y = self.symbolic_function(prior_predictions)

        successes_prior = (y == prior_y).float().mean()

        if true_w is not None:
            explain_acc = 0.
            for i in range(len(true_w)):
                # Get beam search prediction of w, compare to ground truth w
                explain_acc += (result.final_state.w[i][:, 0] == true_w[i]).float().mean()
            explain_acc /= len(true_w)

            prior_acc = (prior_predictions == torch.stack(true_w, 1)).float().mean()

            return torch.mean(successes), successes_prior, explain_acc, prior_acc
        return torch.mean(successes), successes_prior, None, None

    @abstractmethod
    def initial_state(self, P: torch.Tensor, y: Optional[torch.Tensor] = None, w: Optional[torch.Tensor] = None,
                      generate_w=True) -> ST:
        assert not (y is None and w is not None)
        pass

    @abstractmethod
    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam: bool = False) -> torch.Tensor:
        """
        Returns the _probability_ of success. Should probably return the most likely result and compare this instead.
        # TODO: Use a beam search here somehow to parse y
        """
        pass
