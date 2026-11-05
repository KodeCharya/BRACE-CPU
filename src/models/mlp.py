"""Lightweight MLP neural PDE surrogate (CPU-only, FP32).

Predicts the next state u(t+dt) from the current state u(t), the physical
parameter(s), and the time. Designed to be small (~10K-100K parameters) so it
fits the 8 GiB CPU-only constraint.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class MLPSurrogate(nn.Module):
    """Multi-layer perceptron mapping (u, params, t) -> u_next.

    Input dimension: nx + n_params + n_time
    Output dimension: nx
    """

    def __init__(
        self,
        nx: int,
        n_params: int = 1,
        n_time: int = 1,
        hidden_layers: int = 3,
        hidden_units: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        self.nx = int(nx)
        self.n_params = int(n_params)
        self.n_time = int(n_time)
        self.input_dim = self.nx + self.n_params + self.n_time
        self.output_dim = self.nx

        act = _get_activation(activation)

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(act)
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, u: torch.Tensor, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u: (B, nx), params: (B, n_params), t: (B, n_time) -> (B, nx)."""
        x = torch.cat([u, params, t], dim=-1)
        return self.net(x)

    def predict(self, u: torch.Tensor, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convenience forward (no gradient by default in eval mode)."""
        return self.forward(u, params, t)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


def build_model(nx: int, n_params: int = 1, n_time: int = 1,
                hidden_layers: int = 3, hidden_units: int = 64,
                activation: str = "relu") -> MLPSurrogate:
    return MLPSurrogate(
        nx=nx, n_params=n_params, n_time=n_time,
        hidden_layers=hidden_layers, hidden_units=hidden_units,
        activation=activation,
    )
