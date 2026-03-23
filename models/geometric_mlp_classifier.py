#!/usr/bin/env python3
"""
Geometric MLP Classifier - Simple ResidualBlock-based MLP
for three-class behavior classification using geometric features.

Replaces MoE architecture with a straightforward residual MLP.
"""

import torch
import torch.nn as nn
from typing import List


class ResidualBlock(nn.Module):
    """Residual block with projection shortcut, BatchNorm and SiLU."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if out_dim > 1 else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        out = self.fc(x)
        if not isinstance(self.bn, nn.Identity):
            out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + res


class GeometricMLPClassifier(nn.Module):
    """
    Simple residual MLP classifier for geometric features.

    Architecture:
        Input [B, input_dim]
        → ResidualBlock(input_dim → hidden_dims[0])
        → ResidualBlock(hidden_dims[0] → hidden_dims[1])
        → ...
        → Linear(hidden_dims[-1] → num_classes)
        Output [B, num_classes]
    """

    def __init__(self,
                 input_dim: int = 10,
                 num_classes: int = 3,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Build residual MLP
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, dim, dropout=dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Weight initialization
        self._init_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GeometricMLPClassifier: {input_dim}D → {hidden_dims} → {num_classes}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input geometric features [B, input_dim]

        Returns:
            logits: Class logits [B, num_classes]
        """
        return self.network(x)
