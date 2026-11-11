"""Fourier Neural Operator (1D and 2D) with residual (increment) output.

The surrogate predicts the one-save-interval increment

    u_{t+1} = u_t + g_theta(u_t)

which keeps the identity path exact and makes the learned field a correction
rather than a full reconstruction. This is the primary BRACE surrogate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.einsum("bix,iox->box", x_ft[..., :m], self.weight[..., :m])
        out_ft = F.pad(out_ft, (0, n // 2 + 1 - m))
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft = torch.einsum("bixy,ioxy->boxy", x_ft[..., :m1, :m2],
                              self.weight[..., :m1, :m2])
        out_ft = F.pad(out_ft, (0, w // 2 + 1 - m2, 0, h - m1))
        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))


class FNO1d(nn.Module):
    def __init__(self, width: int = 32, modes: int = 16, layers: int = 4, delta_scale: float = 0.1):
        super().__init__()
        self.lift = nn.Conv1d(1, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1)
        )
        self.delta_scale = nn.Parameter(torch.tensor(delta_scale))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = u.unsqueeze(1) if u.dim() == 2 else u
        h = self.lift(x)
        for sp, pw in zip(self.spectral, self.pointwise):
            h = F.gelu(sp(h) + pw(h))
        delta = self.project(h)
        return u + self.delta_scale * delta.squeeze(1)


class FNO2d(nn.Module):
    def __init__(self, width: int = 24, modes1: int = 10, modes2: int = 10,
                 layers: int = 4, delta_scale: float = 0.1):
        super().__init__()
        self.lift = nn.Conv2d(1, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)]
        )
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1)
        )
        self.delta_scale = nn.Parameter(torch.tensor(delta_scale))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = u.unsqueeze(1) if u.dim() == 3 else u
        h = self.lift(x)
        for sp, pw in zip(self.spectral, self.pointwise):
            h = F.gelu(sp(h) + pw(h))
        delta = self.project(h)
        return x.squeeze(1) + self.delta_scale * delta.squeeze(1)


class DeepONet1d(nn.Module):
    """Plain DeepONet baseline: branch(field) . trunk(x) + identity path."""

    def __init__(self, n_points: int = 128, p: int = 64, hidden: int = 96,
                 delta_scale: float = 0.1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_points, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p),
        )
        self.delta_scale = nn.Parameter(torch.tensor(delta_scale))
        self.register_buffer("grid", torch.linspace(0, 2 * torch.pi, n_points)[:, None])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b = self.branch(u)
        t = self.trunk(self.grid)
        return u + self.delta_scale * (b @ t.T)


def build_model(name: str, n_points: int = 128, **kw) -> nn.Module:
    if name == "fno1d":
        return FNO1d(**kw)
    if name == "fno2d":
        return FNO2d(**kw)
    if name == "deeponet1d":
        return DeepONet1d(n_points=n_points, **kw)
    raise ValueError(name)
