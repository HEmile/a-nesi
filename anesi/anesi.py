from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Tuple, Optional, Union
import torch
from torch import nn, Tensor

from inference_models import InferenceModelBase, ST, InferenceResult
from fit_dirichlet import fit_dirichlet
from torch.distributions import Categorical, Dirichlet

EPS = 1e-8

@dataclass
class TrainResult:
    percept_loss: torch.Tensor
    q_loss: torch.Tensor
    entropy: Optional[torch.Tensor]
    P: Tensor

class ANeSIBase(ABC, Generic[ST], nn.Module):

    def __init__(self,
                 q: InferenceModelBase[ST],
                 perception: nn.Module,
                 amount_samples: int,
                 belief_size: List[int],
                 dirichlet_init: float = 0.01,
                 dirichlet_iters: int = 50,
                 dirichlet_lr: float = 1.0,
                 dirichlet_L2: float = 0.0,
                 fixed_alpha: Optional[Union[float, Tensor]] = None,
                 K_beliefs: int = 100,
                 predict_only: bool = False,
                 P_source: str = "prior",
                 q_lr=1e-3,
                 q_loss="mse",
                 policy: str = "off",
                 perception_lr=1e-3,
                 perception_loss='log-q',
                 percept_loss_pref=1.0,
                 entropy_weight: Optional[float] = None,
                 **kwargs
                 ):
        """
        :param q: The inference model
        :param perception: The perception network. Should accept samples from data
        :param amount_samples: The amount of samples to draw to train the inference model
        :param initial_concentration: The initial concentration of the Dirichlet distribution
        :param K_beliefs: The amount of beliefs to keep to fit the Dirichlet
        :param percept_loss_pref: When using perception_loss='both', this will prefer log-q if > 1.0, otherwise sampled
        :param predict_only: If True, assume only the prediction model is used, not the explain model
        :param P_source: If 'prior', use the prior to sample P. If 'percept', use perceptions of P. If 'both', use both. 'percept' is not recommended and is unlikey to converge
        :param q_lr: The learning rate of the inference model
        :param q_loss: The loss function of the inference model. Either 'mse' or 'bce'
        :param policy: The policy to use. Either 'off', 'on' or 'both'. 'on' requires predict_only=False
        """
        super().__init__()
        assert perception_loss in ['sampled', 'log-q', 'both', 'none']
        assert q_loss in ['mse', 'bce']
        assert policy in ['both', 'off', 'on']
        assert P_source in ['prior', 'percept', 'both']
        # prediction only option is only supported for off-policy learning of nrm
        assert not (predict_only and policy in ['on', 'both'])
        # Sampled loss requires explain model
        assert not (predict_only and perception_loss in ['sampled', 'both'])

        self.predict_only = predict_only
        self.P_source = P_source
        self.q = q
        self.perception = perception
        self.amount_samples = amount_samples
        self.belief_size = belief_size
        self.K_beliefs = K_beliefs
        self.dirichlet_iters = dirichlet_iters
        self.dirichlet_lr = dirichlet_lr
        self.dirichlet_L2 = dirichlet_L2
        self.fixed_alpha = None
        self.q_loss = q_loss
        self.policy = policy
        self.perception_loss = perception_loss
        self.percept_loss_pref = percept_loss_pref
        self.entropy_weight = entropy_weight

        # We're training these two models separately, so let's also use two different optimizers.
        #  This ensures we won't accidentally update the wrong model.
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=q_lr)
        if self.perception is not None:
            self.perception_optimizer = torch.optim.Adam(self.perception.parameters(), lr=perception_lr)

        if fixed_alpha is not None:
            if isinstance(fixed_alpha, float):
                self.alpha = nn.Parameter(torch.full((len(belief_size), max(belief_size)), fixed_alpha, requires_grad=False))
            else:
                self.alpha = nn.Parameter(fixed_alpha)
            self.fixed_alpha = self.alpha
        else:
            _x = torch.tensor(dirichlet_init)
            t_initial_concentration = _x + torch.log(-torch.expm1(-_x))
            self.alpha = nn.Parameter(
                torch.full((len(belief_size), max(belief_size)), t_initial_concentration, requires_grad=True))
            self.dirichlet_optimizer = torch.optim.Adam([self.alpha], lr=dirichlet_lr)
        self.beliefs = None

    def joint_matching_loss(self, P: torch.Tensor, q: [torch.Tensor], y: torch.Tensor, w: torch.Tensor,
                            predict_only: bool=False, assume_w_correct: bool=False, punish_incorrect_weight: Optional[float] = 1000.):
        log_q = 0.
        # TODO: This condition looks hacky
        if all([r.shape[-1] == q[0].shape[-1] for r in q]):
            log_q = (torch.stack(q, -1) + EPS).log().sum(tuple(range(1, len(q[0].shape) + 1)))
        else:
            for prob in q:
                if len(prob.shape) == 2:
                    log_q += (prob + EPS).log().sum(-1)
                else:
                    log_q += (prob + EPS).log()

        if predict_only:
            # KL div loss. Increases probability of observed ys
            return -log_q.mean()

        # Joint matching loss
        # (batch,)
        if w.shape[-1] == 1:
            w = w.squeeze(-1)
        log_p = Categorical(probs=P).log_prob(w).sum(-1)
        if not assume_w_correct:
            log_p += (1 - (self.symbolic_function(w) == y).min(-1)[0].float()) * -punish_incorrect_weight
        if self.q_loss == 'mse':
            return (log_q - log_p).pow(2).mean()
        elif self.q_loss == 'bce':
            return nn.BCELoss()(log_q.exp(), log_p.exp())
        raise NotImplementedError()

    def off_policy_loss(self, p_P: Dirichlet, batch_P: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 2
        For now assumes all w_i are the same size
        """
        use_prior = self.P_source != 'percept'
        if use_prior:
            P = p_P.sample((self.amount_samples,))
            if self.P_source == 'both':
                P = torch.cat([P, batch_P], 0)
        else:
            P = batch_P
        p_w = Categorical(probs=P)
        # (batch, |W|)
        w = p_w.sample()

        # (batch,)
        y = self.symbolic_function(w)

        initial_state = self.initial_state(P, y, w, generate_w=not self.predict_only)

        amt_samples = self.amount_samples if use_prior else 1
        result = self.q.forward(initial_state, amt_samples=amt_samples)
        return self.joint_matching_loss(P, result.forward_probabilities, y, w,
                                        predict_only=self.predict_only, assume_w_correct=True)

    def log_q_loss(self, P: torch.Tensor, y: torch.Tensor):
        """
        Perception loss that maximizes the log probability of the label under the NRM model.
        """
        initial_state = self.initial_state(P, y, generate_w=False)
        result = self.q.forward(initial_state)
        stack_ys = (torch.stack(result.forward_probabilities, -1) + EPS).log()
        log_q_y = stack_ys.sum(-1).mean()
        return -log_q_y

    def train_all(self, x: torch.Tensor, y: torch.Tensor) -> TrainResult:
        """
        Algorithm 3
        - Calls algorithm 2 (nrm_loss)
        - Minimizes nrm loss using first optimizer
        - Computes q(y|P)
        - Maximize log prob
        """
        use_off_policy_loss = not self.policy == 'on'
        self.q_optimizer.zero_grad()

        q_loss = 0.
        P = self.perception(x)
        if use_off_policy_loss or self.perception_loss == 'both':
            if self.beliefs is None:
                self.beliefs = P
            else:
                self.beliefs = torch.cat((self.beliefs, P), dim=0)
                if self.beliefs.shape[0] > self.K_beliefs:
                    self.beliefs = self.beliefs[-self.K_beliefs:]

            beliefs = self.beliefs.detach()
            if self.fixed_alpha is not None:
                p_P = Dirichlet(self.alpha)
            else:
                p_P = fit_dirichlet(beliefs, self.alpha, self.dirichlet_optimizer, self.dirichlet_iters, self.dirichlet_L2)

            if use_off_policy_loss:
                _q_loss = self.off_policy_loss(p_P, P.detach())
                _q_loss.backward()
                q_loss += _q_loss
                self.q_optimizer.step()
                self.q_optimizer.zero_grad()

        use_sampled_loss = self.perception_loss in ['sampled', 'both']
        use_log_q_loss = self.perception_loss in ['log-q', 'both']
        use_on_policy_loss = not self.policy == 'off'

        loss_sampled = 0.
        loss_log_q = 0.

        self.perception_optimizer.zero_grad()
        if use_sampled_loss or use_on_policy_loss:
            _loss_sampled, _q_loss = self.sampled_loss(P, y.long(), use_sampled_loss, use_on_policy_loss)
            if use_sampled_loss:
                loss_sampled = _loss_sampled
            if use_on_policy_loss:
                _q_loss.backward()
                q_loss += _q_loss
                self.q_optimizer.step()
        if use_log_q_loss:
            loss_log_q = self.log_q_loss(P, y.long())

        loss_percept = torch.tensor(0., device=x.device)
        entropy = torch.tensor(0., device=x.device)
        if self.perception_loss != 'none':
            w_sampled = 1. if use_sampled_loss else 0.
            w_log_q = 1. if use_log_q_loss else 0.
            loss_percept = w_sampled * loss_sampled + w_log_q * loss_log_q

            # The entropy of the categorical distribution P
            entropy = -torch.sum(P * torch.log(P + EPS), dim=-1).mean()
            if self.entropy_weight:
                # This **maximizes** the entropy. Use a negative entropy weight to minimize
                (loss_percept - self.entropy_weight * entropy).backward()
            else:
                loss_percept.backward()
        self.perception_optimizer.step()

        return TrainResult(q_loss=q_loss, percept_loss=loss_percept, entropy=entropy, P=P)

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
        initial_state = self.initial_state(P, generate_w=not self.predict_only)
        result: InferenceResult[ST] = self.q.beam(initial_state, beam_size=self.amount_samples)
        successes = self.success(result.final_state.y, y, beam=True).float()

        prior_predictions = torch.argmax(P, dim=-1)
        prior_y = self.symbolic_function(prior_predictions)

        successes_prior = (y == prior_y).float().mean()

        if true_w is not None:
            explain_acc = torch.tensor(0., device=successes.device)
            if not self.predict_only:
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
    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam: bool = False) -> torch.Tensor:
        """
        Returns the _probability_ of success. Should probably return the most likely result and compare this instead.
        # TODO: Use a beam search here somehow to parse y
        """
        pass

    def sampled_loss(self, P: torch.Tensor, y: torch.Tensor, compute_perception_loss: bool, compute_q_loss: bool) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        initial_state = self.initial_state(P.detach(), y, generate_w=True)
        result = self.q.forward(initial_state, amt_samples=1)

        percept_loss = q_loss = 0.

        # Take sum of log probs over all dimensions
        log_p_w = result.final_state.log_p_world()

        if compute_perception_loss:
            # # TODO: This might underflow
            # q_y = torch.stack(
            #     result.forward_probabilities[:len(result.final_state.y)], 1
            # ).prod(-1).detach()
            prediction = self.symbolic_function(torch.stack(result.final_state.w, -1))
            # TODO: Not sure about the squeeze here
            # TODO: Also not sure about the min here...
            successes = torch.min(prediction.squeeze() == y, dim=-1)[0].float()
            # print(y)
            if self.q.prune and self.q.no_actions_behaviour == 'raise':
                # If we prune, we know we are successful by definition
                assert successes.all()
                percept_loss = -log_p_w.mean()
            else:
                percept_loss = -(log_p_w * successes).mean() if successes.any() else 0.

        if compute_q_loss:
            log_q_y = (torch.stack(result.forward_probabilities[:len(result.final_state.y)], 1) + EPS).log().sum(-1)
            log_q_y = log_q_y.unsqueeze(-1)
            log_q_w_y = (torch.stack(result.forward_probabilities[len(result.final_state.y):], -1) + EPS).log().sum(-1)
            log_q = log_q_y + log_q_w_y
            log_p_w = log_p_w.detach()
            if self.q_loss == 'mse':
                q_loss = (log_q - log_p_w).pow(2).mean()
            elif self.q_loss == 'bce':
                q_loss = nn.BCELoss()(log_q.exp(), log_p_w.exp())
            else:
                raise NotImplementedError()

        return percept_loss, q_loss
