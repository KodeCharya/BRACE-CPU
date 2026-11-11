"""Shared spectral helpers for periodic PDE benchmarks.

All solvers operate on real-valued fields on a periodic box
``[0, 2*pi)^d`` with Fourier pseudo-spectral spatial discretisation.
Dealiasing uses the standard 2/3 rule.
"""
from __future__ import annotations

import numpy as np


def wavenumbers_1d(n: int, length: float = 2.0 * np.pi) -> np.ndarray:
    """Signed Fourier wavenumbers for an even-length real FFT grid."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    return k.astype(np.float64)


def wavenumbers_2d(n: int, length: float = 2.0 * np.pi):
    """Signed 2D wavenumber grids for ``rfft2`` output (shape n x n//2+1)."""
    kx = (2.0 * np.pi * np.fft.fftfreq(n, d=length / n)).astype(np.float64)
    ky = (2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)).astype(np.float64)
    return kx[:, None], ky[None, :]


def dealias_mask_1d(n: int, cutoff: float = 2.0 / 3.0) -> np.ndarray:
    k = np.abs(wavenumbers_1d(n))
    kmax = k.max()
    return (k <= cutoff * kmax).astype(np.float64)


def dealias_mask_2d(n: int, cutoff: float = 2.0 / 3.0) -> np.ndarray:
    kx, ky = wavenumbers_2d(n)
    kmax = max(np.abs(kx).max(), np.abs(ky).max())
    return ((np.abs(kx) <= cutoff * kmax) & (np.abs(ky) <= cutoff * kmax)).astype(np.float64)


def grad_1d(u: np.ndarray, dx: float) -> np.ndarray:
    """Spectral first derivative with 2/3 dealiasing."""
    n = u.shape[-1]
    k = wavenumbers_1d(n)
    uh = np.fft.fft(u, axis=-1) * dealias_mask_1d(n)
    return np.fft.ifft(1j * k * uh, axis=-1).real


def laplacian_1d(u: np.ndarray) -> np.ndarray:
    n = u.shape[-1]
    k = wavenumbers_1d(n)
    uh = np.fft.fft(u, axis=-1)
    return np.fft.ifft(-(k**2) * uh, axis=-1).real


def grad_2d(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = u.shape[-1]
    kx, ky = wavenumbers_2d(n)
    uh = np.fft.rfft2(u) * dealias_mask_2d(n)
    ux = np.fft.irfft2(1j * kx * uh, s=u.shape).real
    uy = np.fft.irfft2(1j * ky * uh, s=u.shape).real
    return ux, uy


def laplacian_2d(u: np.ndarray) -> np.ndarray:
    n = u.shape[-1]
    kx, ky = wavenumbers_2d(n)
    uh = np.fft.rfft2(u)
    return np.fft.irfft2(-(kx**2 + ky**2) * uh, s=u.shape).real


def grf_1d(
    n: int,
    rng: np.random.Generator,
    length_scale: float = 0.5,
    amplitude: float = 1.0,
    mean: float = 0.0,
    cutoff: float = 2.0 / 3.0,
) -> np.ndarray:
    """Periodic Gaussian random field with a squared-exponential spectrum."""
    k = wavenumbers_1d(n)
    power = np.exp(-(length_scale**2) * k**2)
    power = power * (np.abs(k) <= cutoff * np.abs(k).max())
    coeff = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power)
    coeff[0] = rng.standard_normal() * np.sqrt(power[0])
    field = np.fft.ifft(coeff).real
    field = field - field.mean()
    std = field.std()
    if std > 1e-12:
        field = field / std
    return mean + amplitude * field


def grf_2d(
    n: int,
    rng: np.random.Generator,
    length_scale: float = 0.35,
    amplitude: float = 1.0,
    mean: float = 0.0,
    cutoff: float = 2.0 / 3.0,
) -> np.ndarray:
    """Periodic 2D Gaussian random field (isotropic squared-exponential)."""
    kx, ky = wavenumbers_2d(n)
    k2 = kx**2 + ky**2
    power = np.exp(-(length_scale**2) * k2)
    kmax = max(np.abs(kx).max(), np.abs(ky).max())
    power = power * ((np.abs(kx) <= cutoff * kmax) & (np.abs(ky) <= cutoff * kmax))
    coeff = (rng.standard_normal(power.shape) + 1j * rng.standard_normal(power.shape)) * np.sqrt(power)
    field = np.fft.irfft2(coeff, s=(n, n))
    field = field - field.mean()
    std = field.std()
    if std > 1e-12:
        field = field / std
    return mean + amplitude * field
