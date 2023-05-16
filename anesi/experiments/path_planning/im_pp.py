import torch
import torchvision
from torch import nn

from anesi import InferenceModelBase
from experiments.path_planning.perception import CombResnet18
from experiments.path_planning.state import SPStatePath


class SmallIM(nn.Module):

    def __init__(self, N, n_classes, outputs):
        super().__init__()
        self.N = N
        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 6, 3), # 6 8 8 -> 6 4 4
            nn.MaxPool2d(2, 2),
            # nn.ReLU(True),
            nn.Conv2d(6, 16, 3),
            # nn.MaxPool2d(2, 2),
            nn.ReLU(True))
        self.classifier = nn.Sequential(
            nn.Linear(16 * (N//4)**2, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, outputs))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * (self.N//4)**2)
        x = self.classifier(x)
        return x


class CombIM(nn.Module):
    def __init__(self, N, in_channels, outputs):
        super().__init__()
        self.enc_size = 9 * 64
        self.model = CombResnet18(N, 1, in_channels=in_channels, only_encode=True)
        self.out = nn.Linear(self.enc_size, outputs)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(-1, self.enc_size)
        x = self.out(x)
        return x


class IMCNN(InferenceModelBase):

    def __init__(self, N, n_classes, model):
        super().__init__()
        self.N = N
        self.n_classes = n_classes
        if model == 'resnet':
            self.model_y = torchvision.models.resnet18(weights=None, num_classes=N * N)

            # Hacking ResNets to expect 'in_channels' input channel (and not three)
            # See https://github.com/nec-research/tf-imle/blob/6efb4185c295f646381eb58f775d9b8ecab6044a/WARCRAFT/maprop/models.py
            del self.model_y.conv1
            self.model_y.conv1 = nn.Conv2d(n_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)

            self.model_w = torchvision.models.resnet18(weights=None, num_classes=N * N * n_classes)

            # Hacking ResNets to expect 'in_channels' input channel (and not three)
            # See https://github.com/nec-research/tf-imle/blob/6efb4185c295f646381eb58f775d9b8ecab6044a/WARCRAFT/maprop/models.py
            del self.model_w.conv1
            self.model_w.conv1 = nn.Conv2d(n_classes + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model == 'small':
            self.model_y = SmallIM(N, n_classes, N * N)
            self.model_w = SmallIM(N, n_classes + 1, N * N * 5)
        elif model == 'comb_resnet':
            self.model_y = CombResnet18(N, 1, in_channels=n_classes)
            self.model_w = CombResnet18(N, 5, in_channels=n_classes + 1)

    def distribution(self, state) -> torch.Tensor:
        # Reshape state to be (batch_size, n_classes, N, N)
        p = state.pw
        p = p.view(-1, self.n_classes, self.N, self.N)
        if not state.finished_generating_y():
            dist = torch.sigmoid(self.model_y(p))
            if not dist.shape[-1] == 1:
                dist = dist.unsqueeze(-1)
            return dist
        y = state.y[0]
        y = y.view(-1, 1, self.N, self.N)
        p_y = torch.cat([p, y], dim=1)
        dist = torch.softmax(torch.reshape(self.model_w(p_y), (-1, self.N * self.N, self.n_classes)), dim=-1)
        return dist


class IMPath(InferenceModelBase):

    def __init__(self, N, n_classes, model):
        super().__init__(prune=True)
        self.N = N
        self.n_classes = n_classes
        self.n_directions = 8
        if model == 'path_resnet':
            self.model_y = torchvision.models.resnet18(weights=None, num_classes=self.n_directions)

            # Hacking ResNets to expect 'in_channels' input channel (and not three)
            # See https://github.com/nec-research/tf-imle/blob/6efb4185c295f646381eb58f775d9b8ecab6044a/WARCRAFT/maprop/models.py
            del self.model_y.conv1
            self.model_y.conv1 = nn.Conv2d(self.n_classes + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            self.model_w = torchvision.models.resnet18(weights=None, num_classes=N * N * n_classes)

            # Hacking ResNets to expect 'in_channels' input channel (and not three)
            # See https://github.com/nec-research/tf-imle/blob/6efb4185c295f646381eb58f775d9b8ecab6044a/WARCRAFT/maprop/models.py
            del self.model_w.conv1
            self.model_w.conv1 = nn.Conv2d(n_classes + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model == 'path_small':
            self.model_y = SmallIM(N, n_classes + 1, self.n_directions)
            self.model_w = SmallIM(N, n_classes + 1, N * N * 5)
        elif model == 'path_comb_resnet':
            self.model_y = CombIM(N, n_classes + 1, self.n_directions)
            self.model_w = CombIM(N, n_classes + 1, self.n_classes)

    def distribution(self, state: SPStatePath) -> torch.Tensor:
        # Reshape state to be (batch_size, n_classes, N, N)
        p = state.pw
        p = p.view(-1, self.n_classes, self.N, self.N)

        if not state.finished_generating_y():
            if state.parallel_y:
                one_hot_prev_y = nn.functional.one_hot(state.constraint_coords, self.N * self.N).float()
                one_hot_prev_y = one_hot_prev_y.reshape(-1, 1, self.N, self.N)
                p = p.unsqueeze(1).repeat(1, state.constraint_coords.shape[-1], 1, 1, 1)
                p = p.reshape(-1, self.n_classes, self.N, self.N)
                p = torch.cat([p, one_hot_prev_y], dim=1)
                dist = torch.softmax(self.model_y(p), dim=-1).reshape(-1, state.constraint_coords.shape[-1], self.n_directions)
                dist = torch.cat([dist, torch.ones((dist.shape[0], dist.shape[1], 1), device=dist.device)], dim=-1)
                return dist
            else:
                dist = torch.zeros((p.shape[0], self.n_directions + 1), device=p.device)
                last_y = state.prev_ys[..., -1]
                mask = last_y < self.N * self.N - 1

                # Index 8 represents the 'do nothing' action
                dist[~mask, self.n_directions] = 1.

                one_hot_y = nn.functional.one_hot(last_y[mask], self.N * self.N).float().reshape(-1, 1, self.N, self.N)
                p = torch.cat([p[mask], one_hot_y], dim=1)
                dist[mask, :self.n_directions] = torch.softmax(self.model_y(p), dim=-1)
                return dist
        raise NotImplementedError()

# class PSIMTransformer(InferenceModelBase):
#
#     def __init__(self, N, d_model, nhead, num_layers, num_classes):
#         super().__init__()
#         # IDEA: Use just a CNN, not autoregressive (just predict whether it is part of the path or not)
#         # Of course not very reliable/ consistent but whatever. It will do.
#         # Can probably just use the same ResNEt
#         self.grid_dim = N
#         self.d_model = d_model
#         self.enc_p = nn.Linear(N * N * num_classes, self.d_model)
#         self.y_embedding = nn.Embedding(num_classes, self.d_model)
#         self.positional_embedding = nn.Embedding(N * N, self.d_model)
#         self.transformer = nn.Transformer(
#             self.d_model,
#             nhead,
#             num_layers,
#             dim_feedforward=self.d_model * 4,
#             activation='relu'
#         )
#         self.fc = nn.Linear(self.d_model, num_classes)
#
#     def distribution(self, state) -> torch.Tensor:
#         P = state.oh_pw
#         encoded_p = self.enc_p(P)
#         # Input shape: (batch_size, grid_dim * grid_dim)
#         batch_size = x.shape[0]
#         positions = torch.arange(0, self.grid_dim * self.grid_dim, device=x.device).unsqueeze(0).repeat(batch_size, 1)
#
#         x = self.y_embedding(x) + self.positional_embedding(positions)
#
#         # Required shape for transformer: (S, N, E) where S is the sequence length, N is the batch size, and E is the embedding dimension
#         x = x.transpose(0, 1)
#
#         x = self.transformer(x)
#
#         x = x.transpose(0, 1)
#
#         x = self.fc(x)
#
#         return x