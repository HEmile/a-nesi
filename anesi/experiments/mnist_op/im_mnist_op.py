from typing import List, Optional

from anesi import InferenceModelBase
from experiments.mnist_op import MNISTState
import torch
import torch.nn as nn

from inference_models import InferenceResult


class InferenceModelMnist(InferenceModelBase[MNISTState]):

    def __init__(self, N: int, output_dims: List[int], layers = 1, hidden_size: int = 200, prune: bool = True):
        super().__init__(prune)
        self.N = N
        self.layers = layers
        self.output_dims = output_dims

        # w_encoding_len = 10 if w_encoding == "base10" else 4
        input_queries = [nn.Linear(2 * 10 * N + sum(output_dims[:i]), hidden_size) for i in range(len(output_dims))]
        output_queries = [nn.Linear(hidden_size, dim) for dim in output_dims]

        y_size = sum(output_dims)

        self.input_layer = nn.ModuleList(input_queries +
                                     [nn.Linear(20 * N + y_size + i * 10, hidden_size) for i in range(2 * N)])
        self.hiddens = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range((2 * N + len(output_dims)) * (layers - 1))])
        self.outputs = nn.ModuleList(output_queries +
                                     [nn.Linear(hidden_size, 10) for _ in range(2 * N)])

    def distribution(self, state: MNISTState) -> torch.Tensor:
        p = state.probability_vector()
        layer_index = len(state.oh_state)
        inputs = torch.cat([p] + state.oh_state, -1)

        z = torch.relu(self.input_layer[layer_index](inputs))

        for i in range(self.layers - 1):
            z = torch.relu(self.hiddens[i * (2 * self.N + len(self.output_dims)) + layer_index](z))

        logits = self.outputs[layer_index](z)
        if logits.shape[-1] > 1:
            dist = torch.softmax(logits, -1)
            return dist
        dist = torch.sigmoid(logits)
        return dist

class IndependentIMMnist(InferenceModelBase[MNISTState]):

    def __init__(self, N: int, output_dims: List[int], layers = 1, hidden_size: int = 200):
        super().__init__(False)
        self.layers = layers
        self.output_dims = output_dims

        y_len = sum(output_dims)

        # w_encoding_len = 10 if w_encoding == "base10" else 4
        self.input_query = nn.Linear(2 * 10 * N, hidden_size)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)])
        self.output_query = nn.Linear(hidden_size, y_len)

    def distribution(self, state: MNISTState) -> torch.Tensor:
        assert not state.generate_w
        if len(state.oh_state) == 0:
            p = state.probability_vector()
            z = torch.relu(self.input_query(p))
            for i in range(self.layers - 1):
                z = torch.relu(self.hiddens[i](z))
            self.logits = self.output_query(z)
        i = len(state.oh_state)
        dim = self.output_dims[i]
        seen_so_far = sum(self.output_dims[:i])
        if dim > 1:
            dist = torch.softmax(self.logits[..., seen_so_far:seen_so_far + dim], -1)
            return dist
        dist = torch.sigmoid(self.logits[..., seen_so_far: seen_so_far + dim])
        return dist

    def beam(self, initial_state: MNISTState, beam_size: int):
        state = initial_state
        for _ in range(len(self.output_dims)):
            dist = self.distribution(state)
            probs, action = torch.max(dist, -1)
            state = state.next_state(action)
        return InferenceResult(state, None, None, None, None)

