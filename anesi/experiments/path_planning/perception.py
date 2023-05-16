from math import sqrt

import torchvision
from torch import nn
import torch


class SmallNet(nn.Module):
    def __init__(self, N=5):
        super(SmallNet, self).__init__()
        self.N = N
        self.size = 24
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5), # 6 8 8 -> 6 4 4
            nn.MaxPool2d(2, 2),  # 6 4 4 -> 6 2 2
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.size, 84),
            nn.ReLU(True),
            nn.Linear(84, N),
        )

    def forward(self, x):
        """
        Assuming x is of shape [b, ds, 28, 28] where ds is the number of digits
        """

        batch_size = x.shape[0]
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        x = x.reshape(batch_size, -1, self.N)
        return x

class MLP(nn.Module):
    def __init__(self, N=5):
        super().__init__()
        self.N = N
        self.input_size = 3 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, N),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape(batch_size, -1, self.N)
        return x

def get_resnet(amt_classes):
    model = torchvision.models.resnet18(weights=None, num_classes=amt_classes)

    return model

class SPPerception(nn.Module):

    def __init__(self, N, n_classes, model="small"):
        super().__init__()
        if model == 'small':
            self.model = SmallNet()
        elif model == 'mlp':
            self.model = MLP()
        else:
            self.model = get_resnet(amt_classes=n_classes)
        self.N = N
        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch_size, channels, N * 8, N * 8)
        # We need (batch_size * N * N, channels, 8, 8)
        # Output: (batch_size, N * N, n_classes)
        channels = x.shape[1]
        cell_size_x = x.shape[2] // self.N
        cell_size_y = x.shape[3] // self.N
        # Complicated reshape procedure to ensure the cells are properly tiled
        x = x.reshape(-1, channels, self.N, cell_size_x, self.N, cell_size_y)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(-1, channels, cell_size_x, cell_size_y)
        P = self.model(x)
        P = P.reshape(-1, self.N * self.N, self.n_classes)
        return torch.softmax(P, dim=-1)

class CombResnet18(nn.Module):
    def __init__(self, N, n_classes, in_channels=3, only_encode=False):
        super().__init__()
        out_features = N * N * n_classes
        self.n_classes = n_classes
        self.resnet_model = torchvision.models.resnet18(weights=None, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Ensures the output of the resnet is b x 64 x N x N
        self.pool = nn.AdaptiveMaxPool2d((N, N))
        # Shared output layer for all cells
        self.only_encode = only_encode
        if not only_encode:
            self.out = nn.Linear(64, n_classes)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        if self.only_encode:
            return x
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.out(x)
        x = torch.softmax(x, dim=-1)
        x = x.reshape(x.shape[0], -1, self.n_classes)
        return x
