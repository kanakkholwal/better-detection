"""PyTorch 1D-ECG models: SimpleCNN, Inception-style, ResNet-style.

Every model exposes `.penultimate(x)` returning the pre-head embedding computed with
no dropout. Callers must still use `model.eval()` + `torch.no_grad()` to keep
BatchNorm statistics in eval mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EMBEDDING_DIM, INPUT_LENGTH, NUM_LEADS


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = NUM_LEADS, emb_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.penult = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(emb_dim, 1)
        self.apply(_init_weights)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        return self.penult(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.penultimate(x)
        z = self.dropout(z)
        return self.head(z).squeeze(-1)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_per_branch: int = 32):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=1),
            nn.BatchNorm1d(out_per_branch),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_per_branch),
            nn.ReLU(inplace=True),
        )
        self.b5 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_per_branch),
            nn.ReLU(inplace=True),
        )
        self.bp = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_per_branch, kernel_size=1),
            nn.BatchNorm1d(out_per_branch),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_per_branch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class InceptionNet(nn.Module):
    def __init__(self, in_channels: int = NUM_LEADS, emb_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.inc1 = InceptionBlock(32, 32)   # -> 128
        self.pool1 = nn.MaxPool1d(2)
        self.inc2 = InceptionBlock(128, 48)  # -> 192
        self.pool2 = nn.MaxPool1d(2)
        self.inc3 = InceptionBlock(192, 64)  # -> 256
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.penult = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(emb_dim, 1)
        self.apply(_init_weights)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.inc1(x))
        x = self.pool2(self.inc2(x))
        x = self.gap(self.inc3(x))
        return x

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        return self.penult(self._trunk(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.penultimate(x)
        z = self.dropout(z)
        return self.head(z).squeeze(-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int = NUM_LEADS, emb_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock1D(64, 64), ResidualBlock1D(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock1D(64, 128, stride=2), ResidualBlock1D(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock1D(128, 256, stride=2), ResidualBlock1D(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock1D(256, 512, stride=2), ResidualBlock1D(512, 512))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.penult = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(emb_dim, 1)
        self.apply(_init_weights)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.gap(x)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        return self.penult(self._trunk(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.penultimate(x)
        z = self.dropout(z)
        return self.head(z).squeeze(-1)


def build_model(name: str) -> nn.Module:
    name_l = name.lower()
    if name_l in ("simplecnn", "simple_cnn", "cnn"):
        return SimpleCNN()
    if name_l in ("inception", "inceptionnet"):
        return InceptionNet()
    if name_l in ("resnet", "resnet1d"):
        return ResNet1D()
    raise ValueError(f"Unknown model: {name}")
