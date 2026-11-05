"""Data generation package."""

from .dataset import (
    generate_burgers_trajectory,
    generate_advection_diffusion_trajectory,
    generate_burgers_dataset,
    random_fourier_initial_condition,
    gaussian_random_field_initial_condition,
    save_trajectory,
    sha256_file,
    TrajectoryMetadata,
)

__all__ = [
    "generate_burgers_trajectory",
    "generate_advection_diffusion_trajectory",
    "generate_burgers_dataset",
    "random_fourier_initial_condition",
    "gaussian_random_field_initial_condition",
    "save_trajectory",
    "sha256_file",
    "TrajectoryMetadata",
]
