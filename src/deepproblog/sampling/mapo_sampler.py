from typing import Optional

import torch

import storch
from deepproblog.sampling.memoizer import Memoizer
from storch.sampling import SampleWithoutReplacement
from storch.sampling.seq import AncestralPlate


class MemoryAugmentedDPLSampler(SampleWithoutReplacement):
    """
    Implements a variant of MAPO: https://arxiv.org/abs/1807.02322 specific for DeepProbLog
    Reuses the Stochastic Beam Search for efficient computation as suggested in https://www.jmlr.org/papers/v21/19-985.html
    """
    def __init__(self, plate_name: str, k: int, memoizer: Memoizer):
        """

        Args:
            plate_name:
            k: Total amount of samples. k-1 will be summed over (highest probability samples), then 1 reinforce sample
        """
        super().__init__(plate_name, k)
        self.memoizer: Memoizer = memoizer

    def select_samples(
            self, perturbed_log_probs: storch.Tensor, joint_log_probs: storch.Tensor,
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Given the perturbed log probabilities and the joint log probabilities of the new options, select which one to
        use for the sample.
        :param perturbed_log_probs: plates x (k? * |D_yv|). Perturbed log-probabilities. k is present if first_sample.
        :param joint_log_probs: plates x (k? * |D_yv|). Joint log probabilities of the options. k is present if first_sample.
        :param first_sample:
        :return: perturbed log probs of chosen samples, joint log probs of chosen samples, index of chosen samples
        """

        # TODO:
        #  l proofs in memoizer
        #  Choose those l proofs to sum over
        #  Sample k-l from the rest without replacement

        amt_samples = min(self.k, perturbed_log_probs.shape[-1])
        joint_log_probs_sum_over, sum_over_index = torch.topk(joint_log_probs, min(self.k - 1, amt_samples), dim=-1)
        perturbed_l_p_sum_over = perturbed_log_probs.gather(dim=-1, index=sum_over_index)

        if amt_samples == self.k:
            # Set the perturbed log probs of the samples chosen through the beam search to a low value
            filtered_perturbed_log_probs = torch.scatter(perturbed_log_probs, -1, sum_over_index, -1e10)
            perturbed_l_p_sample, sample_index = torch.max(filtered_perturbed_log_probs, dim=-1)
            perturbed_l_p_sample, sample_index = perturbed_l_p_sample.unsqueeze(-1), sample_index.unsqueeze(-1)
            joint_log_probs_sample = joint_log_probs.gather(dim=-1, index=sample_index)
            return (torch.cat((perturbed_l_p_sum_over, perturbed_l_p_sample), dim=-1),
                     torch.cat((joint_log_probs_sum_over, joint_log_probs_sample), dim=-1),
                     torch.cat((sum_over_index, sample_index), dim=-1))

        return perturbed_l_p_sum_over, joint_log_probs_sum_over, sum_over_index


    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: AncestralPlate
    ) -> Optional[storch.Tensor]:
        """
        Returns the weighting of the sample
        :param tensor:
        :param plate:
        :return:
        """

        # TODO: I think this just works? Nothing to change here??
        #  Well... If it is k-l, we need to sample the rest without replacement (which is more efficient... easier to find proofs!)
        #  I suppose we can either use the unordered set estimator or Reinforce without replacement here.
        #  Also the l is query-dependent... How does that work?
        amt_samples = plate.log_probs.shape[-1]
        if amt_samples == self.k:
            def _priv(log_probs: torch.Tensor):
                joint_probs = log_probs[..., :-1].exp()
                iw_sample = (1. - joint_probs.sum(dim=-1).detach()).unsqueeze(-1)
                weighting = torch.cat((joint_probs, iw_sample), dim=-1)
                return weighting
            return storch.deterministic(_priv)(plate.log_probs)

        return plate.log_probs.exp()
