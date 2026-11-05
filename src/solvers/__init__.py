"""PDE solver package."""

from .base import PDESolver, periodic_gradient, periodic_laplacian, relative_l2_error, max_abs_error
from .burgers import BurgersSolver
from .advection_diffusion import AdvectionDiffusionSolver

__all__ = [
    "PDESolver",
    "periodic_gradient",
    "periodic_laplacian",
    "relative_l2_error",
    "max_abs_error",
    "BurgersSolver",
    "AdvectionDiffusionSolver",
]
