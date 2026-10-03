
import torch
import torch.nn as nn

class OneLayerCNN(nn.Module):
    def __init__(self, in_ch=3, n_filters=16, k=3, out_features=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_filters, k, padding=k // 2)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.projection = nn.Linear(n_filters * 4, out_features)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return torch.relu(self.projection(x))


class TwoLayerCNN(nn.Module):
    def __init__(self, in_ch=3, n_filters=16, k=3, out_features=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, n_filters, k, padding=k // 2)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, k, padding=k // 2)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.projection = nn.Linear(n_filters * 2 * 4, out_features)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return torch.relu(self.projection(x))