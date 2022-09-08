import random
from typing import Optional, Sequence, List, Dict, Tuple

import torch
import numpy as np

import storch
from deepproblog.query import Query
from deepproblog.sampling.memoizer import Memoizer
from deepproblog.sampling.sampler import Sampler, QueryMapper
from storch import StochasticTensor, CostTensor
from storch.method import Method
from storch.method.multi_sample_reinforce import SeqMethod
from storch.sampling import SampleWithoutReplacement, UnorderedSet
from storch.sampling.seq import AncestralPlate


class MemoryAugmentedDPLSampler(Sampler):
    """
    Implements a variant of MAPO: https://arxiv.org/abs/1807.02322 specific for DeepProbLog
    Reuses the Stochastic Beam Search for efficient computation as suggested in https://www.jmlr.org/papers/v21/19-985.html

    # TODO: Assuming k_swor = 1 for now. If it works well enough, i keep it.
    """
    def __init__(self, k_proofs: int, k_swor: int, memoizer: Memoizer, n_classes_query:int, entropy_weight: float, mapper: QueryMapper=None):
        """

        Args:
            plate_name:
            k: Total amount of samples. k-1 will be summed over (highest probability samples), then 1 reinforce sample
        """
        super().__init__(None, k_proofs + k_swor, n_classes_query, entropy_weight, mapper)

        self.sampler = MAPOSWORSampler(k_proofs, k_swor, memoizer)
        self.method = MemoryAugmentedDPLEstimator("z", self.sampler)

    def sample_atoms(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
        self.sampler.prepare_sampling(queries)
        super().sample_atoms(queries, samples)

    def create_method(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Method:
        return self.method


class MAPOSWORSampler(UnorderedSet):
    """
    Implements a variant of MAPO: https://arxiv.org/abs/1807.02322 specific for DeepProbLog
    Reuses the Stochastic Beam Search for efficient computation as suggested in https://www.jmlr.org/papers/v21/19-985.html
    """
    def __init__(self, k_proofs: int, k_swor:int, memoizer: Memoizer, clipping_threshold: float=0.2):
        """

        Args:
            k_buffer: Amount of proofs to sum over
            k_swor: Amount of samples without replacement
        """
        super().__init__("z", k_proofs + k_swor)  # Note: This fixes the plate_name to z.

        self.k_proofs = k_proofs
        self.k_swor = k_swor
        self.sample_index: int = None
        self.proofs_tensors: List[storch.Tensor] = []
        self.memoizer: Memoizer = memoizer
        self.proofs: List = []
        self.clipping_threshold = clipping_threshold

    def prepare_sampling(self, queries: Sequence[Query]):
        proofs_list: List[List[List[int]]] = list(map(self.memoizer.get_proofs, queries))
        max_amt_proofs = max(map(len, proofs_list))

        self.proofs_tensors = []
        if max_amt_proofs > 0:
            size_proofs = len(next(filter(lambda l: len(l) > 0, proofs_list))[0])
            amt_proofs = min(self.k_proofs, max_amt_proofs) # TODO: Just arbitrary order for now
            for i in range(size_proofs):
                proof_t = np.full((len(queries), amt_proofs), -1, dtype=np.int32)

                for q_i, proofs_for_query in enumerate(proofs_list):
                    if len(proofs_for_query) > self.k_proofs:
                        # TODO: Smarter proof sampling
                        proofs_for_query = random.sample(proofs_for_query, self.k_proofs)
                    for p_i, proof in enumerate(proofs_for_query):
                        if p_i == amt_proofs:
                            break
                        proof_t[q_i, p_i] = proof[i]

                self.proofs_tensors.append(storch.denote_independent(torch.tensor(proof_t), 0, "batch"))

        self.sample_index = 0

    @storch.deterministic
    def _filter_perturbed_log_probs(self, memoized: torch.Tensor, perturbed_log_probs: torch.Tensor) -> torch.Tensor:
        b, k = memoized.shape
        filtered_plp = torch.clone(perturbed_log_probs)
        for query_i in range(b):
            for proof_i in range(k):
                if memoized[query_i, proof_i] == -1:
                    break
                else:
                    filtered_plp[query_i, memoized[query_i, proof_i]] = -1e10
        return filtered_plp

    @storch.deterministic
    def _find_log_probs(self, chosen_samples, joint_log_probs, perturbed_log_probs) -> (torch.Tensor, torch.Tensor):
        # Gathers the log_probs of the chosen samples, making sure that -1 samples are set to -1e10
        b, k = joint_log_probs.shape
        chosen_samples[chosen_samples == -1] = k
        ignore_row = torch.full((b, 1), -1e10)
        joint_log_probs = torch.cat([joint_log_probs, ignore_row], dim=-1)
        perturbed_log_probs = torch.cat([perturbed_log_probs, ignore_row], dim=-1)
        joint_log_probs_sample = joint_log_probs.gather(dim=-1, index=chosen_samples)
        perturbed_log_probs_sample = perturbed_log_probs.gather(dim=-1, index=chosen_samples)

        new_chosen_samples = torch.clone(chosen_samples)
        new_chosen_samples[new_chosen_samples == k] = 0
        return joint_log_probs_sample, perturbed_log_probs_sample, new_chosen_samples


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
        Chosen samples have a certain structure: it starts of with the memoized samples. Since there is a variable amount
        of memoization, this tensor is filled up with 0s. This means the first option gets chosen here!
        The rest of the tensor (k_swor) consists of the samples without replacement.
        """

        # TODO: If max_amt_proofs > self.k_proofs
        #  Then select the top k_proofs proofs by joint_log_probs

        if len(self.proofs_tensors) == 0:
            # Do a regular SWOR when no proofs have been found yet.
            return super().select_samples(perturbed_log_probs, joint_log_probs)

        # batch x k
        # First l_i are memoized samples. The rest is -1
        memoized_samples = self.proofs_tensors[self.sample_index]
        amt_proofs = memoized_samples.shape[-1]

        # Map samples to the corresponding element in the k x |D| indexing.
        # Presumes that the first elements sampled are memoized
        if self.sample_index > 0 and amt_proofs > 1:
            size_d = int(perturbed_log_probs.shape[-1] / (amt_proofs + self.k_swor)) \
                if self.sample_index > 0 \
                else perturbed_log_probs.shape[-1]
            memoized_samples += torch.arange(0, amt_proofs, dtype=torch.long).unsqueeze(0) * size_d * memoized_samples.ne(-1)

        # Get the top-k from the _remaining_ samples. Ie we should filter the perturbed log probs!
        filtered_plp = self._filter_perturbed_log_probs(memoized_samples, perturbed_log_probs)
        # TODO: Can also use self.k instead of self.k_swor, but this requires enabling self.k_swor > 1
        _, top_k_samples = torch.topk(filtered_plp, self.k_swor, dim=-1)

        chosen_samples = storch.cat([memoized_samples, top_k_samples], dim=-1)

        self.sample_index += 1

        return self._find_log_probs(chosen_samples, joint_log_probs, perturbed_log_probs)

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: AncestralPlate
    ) -> Optional[storch.Tensor]:
        """
        Returns the weighting of the sample
        :param tensor:
        :param plate:
        :return:
        """

        # TODO:
        #  Optimization: Do not compute for self.sample_index < len(self.proofs_tensors)
        # TODO:
        #  Find max_amt_samples
        #  If part of max_amt_samples: Use sum-over, but filter wherever it is -1 and set this to probability 0
        #  If not part of max_amt_samples: Use unordered-set-estimator
        # TODO:
        #  Seeing some very high log_probs every now and then? Even if the distribution doesn't reflect this possibility
        if len(self.proofs_tensors) == 0:
            return super().weighting_function(tensor, plate)

        memoized_samples = self.proofs_tensors[0]
        amt_proofs = memoized_samples.shape[-1]

        k_swor = self.k_swor
        clipping_threshold = self.clipping_threshold

        def _priv(log_probs: torch.Tensor, memoized_samples: torch.Tensor) -> torch.Tensor:
            has_proof = memoized_samples.ne(-1)
            joint_probs = log_probs[..., :amt_proofs].exp() * has_proof

            buffer_weight = joint_probs.sum(dim=-1).detach()
            biased_buffer_weight = buffer_weight.clamp(min=clipping_threshold)
            has_any_proof = has_proof.any(dim=-1)
            iw_sum = torch.where(has_any_proof, biased_buffer_weight / buffer_weight, buffer_weight)
            joint_probs_weighted = joint_probs * iw_sum.unsqueeze(1)
            iw_sample = (1 - torch.where(has_any_proof, biased_buffer_weight, buffer_weight)).unsqueeze(-1).expand(-1, k_swor)

            weighting = torch.cat((joint_probs_weighted, iw_sample), dim=-1)
            return weighting
        return storch.deterministic(_priv)(plate.log_probs, memoized_samples._tensor)


class MemoryAugmentedDPLEstimator(SeqMethod):
    def __init__(self, plate_name, sample_method):
        super().__init__(plate_name, sample_method)

    def estimator(
            self, tensor: StochasticTensor, cost: CostTensor
    ) -> Tuple[Optional[storch.Tensor], Optional[storch.Tensor]]:
        # Only uses the last sample for the score function
        zeros = torch.zeros_like(cost, dtype=tensor.dtype)
        cost_plate: AncestralPlate = tensor.get_plate(self.plate_name)

        if cost_plate.n == self.sampling_method.k:
            p_index = tensor.get_plate_dim_index(self.plate_name)
            slize = [slice(None)] * p_index
            slize.append(self.sampling_method.k - 1)
            slize = tuple(slize)
            # Find the correct index for the last sample
            zeros[slize] = cost_plate.log_probs._tensor[slize]

        return zeros, None
