from typing import Optional, Sequence, List, Dict, Tuple

import torch

import storch
from deepproblog.query import Query
from deepproblog.sampling.memoizer import Memoizer
from deepproblog.sampling.sampler import Sampler, QueryMapper
from storch import StochasticTensor, CostTensor
from storch.method import Method
from storch.method.multi_sample_reinforce import SeqMethod
from storch.sampling import SampleWithoutReplacement
from storch.sampling.seq import AncestralPlate


class MemoryAugmentedDPLSampler(Sampler):
    """
    Implements a variant of MAPO: https://arxiv.org/abs/1807.02322 specific for DeepProbLog
    Reuses the Stochastic Beam Search for efficient computation as suggested in https://www.jmlr.org/papers/v21/19-985.html
    """
    def __init__(self, k: int, memoizer: Memoizer, n_classes_query:int, entropy_weight: float, mapper: QueryMapper=None):
        """

        Args:
            plate_name:
            k: Total amount of samples. k-1 will be summed over (highest probability samples), then 1 reinforce sample
        """
        super().__init__(None, k, n_classes_query, entropy_weight, mapper)

        self.sampler = MAPOSWORSampler(k, memoizer)
        self.method = MemoryAugmentedDPLEstimator("z", self.sampler)

    def sample_atoms(self, queries: Sequence[Query], samples: List[Dict[str, torch.Tensor]]):
        self.sampler.prepare_sampling(queries)
        super().sample_atoms(queries, samples)

    def create_method(self, dists: List[torch.Tensor], target_l: List[torch.Tensor]) -> Method:
        return self.method


class MAPOSWORSampler(SampleWithoutReplacement):
    """
    Implements a variant of MAPO: https://arxiv.org/abs/1807.02322 specific for DeepProbLog
    Reuses the Stochastic Beam Search for efficient computation as suggested in https://www.jmlr.org/papers/v21/19-985.html
    """
    def __init__(self, k: int, memoizer: Memoizer):
        """

        Args:
            plate_name:
            k: Total amount of samples. k-1 will be summed over (highest probability samples), then 1 reinforce sample
        """
        super().__init__("z", k)  # Note: This fixes the plate_name to z.

        self.sample_index: int = None
        self.proofs_tensors: List[storch.Tensor] = []
        self.memoizer: Memoizer = memoizer
        self.proofs: List = []

    def prepare_sampling(self, queries: Sequence[Query]):
        proofs_list: List[List[List[int]]] = list(map(self.memoizer.get_proofs, queries))
        max_amt_proofs = max(max(map(len, proofs_list)), self.k)
        size_proofs = len(next(filter(lambda l: len(l) > 0, proofs_list))[0])
        self.proofs_tensors = []
        for i in range(size_proofs):
            proof_t: torch.Tensor = torch.full((len(queries), max_amt_proofs), -1)

            for q_i, proofs_for_query in enumerate(proofs_list):
                for p_i, proof in enumerate(proofs_for_query):
                    proof_t[p_i, q_i] = proof[i]

            self.proofs_tensors.append(storch.denote_independent(proof_t, 0, "batch"))

        self.sample_index = 0

    @storch.deterministic
    def _join(self, memoized: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        # Fill the -1 entries in memoized with the top-k samples
        b, k = memoized.shape
        joined_sample = torch.clone(memoized)
        for query_i in range(b):
            for proof_i in range(k):
                if memoized[query_i, proof_i] == -1:
                    if proof_i != 0:
                        joined_sample[query_i, proof_i:] = samples[query_i, :k-proof_i]
                    break
        return joined_sample

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

        # TODO Implementation:
        #  How to choose the correct current sampling index?

        # TODO:
        #  l proofs in memoizer
        #  Choose those l proofs to sum over
        #  Sample k-l from the rest without replacement

        # batch x k
        # First l_i are memoized samples. The rest is -1
        memoized_samples = self.proofs_tensors[self.sample_index]
        k = memoized_samples.shape[-1]
        size_d = perturbed_log_probs.shape[-1] / k

        # Map samples to the corresponding element in the k x |D| indexing.
        # Presumes that the first elements sampled are memoized
        memoized_samples += torch.range(0, k) * size_d * (memoized_samples != -1)

        # Get the top-k from the _remaining_ samples. Ie we should filter the perturbed log probs!
        filtered_plp = self._filter_perturbed_log_probs(memoized_samples, perturbed_log_probs)
        _, top_k_samples = torch.topk(filtered_plp, k, dim=-1)

        # Merge the top k samples without replacement and the memoized samples
        chosen_samples = self._join(memoized_samples, top_k_samples)

        joint_log_probs_sample = joint_log_probs.gather(dim=-1, index=chosen_samples)
        perturbed_log_probs_sample = perturbed_log_probs.gather(dim=-1, index=chosen_samples)

        self.sample_index += 1
        return joint_log_probs_sample, perturbed_log_probs_sample, chosen_samples

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
