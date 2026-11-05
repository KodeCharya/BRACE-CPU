"""Risk features for neural PDE surrogate failure prediction.

Candidate signals (evaluated independently in Phase 6 / ablation):
1. Normalized PDE residual of the surrogate's one-step prediction.
2. Temporal prediction inconsistency (one step vs two half-steps).
3. Physical constraint deviation (mass conservation for Burgers).
4. Prediction increment magnitude.
5. Local solution gradient magnitude (steeper = harder to predict).

All signals are cheap to compute from the current state, the predicted next
state, the PDE parameters, and the time step -- no future ground truth required.
"""

from __future__ import annotations

import numpy as np

from solvers.base import periodic_gradient, periodic_laplacian


def _dx_from_state(u: np.ndarray, L: float = 1.0) -> float:
    return L / u.shape[0]


# ---------------------------------------------------------------------------
# 1. Normalized PDE residual (Burgers: u_t + u u_x = nu u_xx)
# ---------------------------------------------------------------------------

def burgers_pde_residual(u_t: np.ndarray, u_pred: np.ndarray, nu: float, dt: float, L: float = 1.0) -> np.ndarray:
    """Pointwise residual of the Burgers PDE for the one-step prediction.

    Approximates du/dt ~ (u_pred - u_t)/dt and compares to -u u_x + nu u_xx
    evaluated at u_t. Returns the residual field (nx,).
    """
    dx = _dx_from_state(u_t, L)
    dudt = (u_pred - u_t) / dt
    ux = periodic_gradient(u_t, dx)
    uxx = periodic_laplacian(u_t, dx)
    rhs = -u_t * ux + nu * uxx
    return dudt - rhs


def normalized_residual(u_t: np.ndarray, u_pred: np.ndarray, nu: float, dt: float, L: float = 1.0) -> float:
    res = burgers_pde_residual(u_t, u_pred, nu, dt, L)
    return float(np.linalg.norm(res) / (np.linalg.norm(u_t) + 1e-12))


# ---------------------------------------------------------------------------
# 2. Temporal prediction inconsistency
# ---------------------------------------------------------------------------

def temporal_inconsistency(
    surrogate, u_t: np.ndarray, params: np.ndarray, t: float, dt: float
) -> float:
    """Difference between one full step and two half-step predictions.

    Cheap surrogate-only signal: if the model is locally consistent, one step
    of size dt should roughly equal two steps of size dt/2.
    """
    u_full = surrogate.predict(u_t, params, np.array([t + dt]))
    u_half = surrogate.predict(u_t, params, np.array([t + dt / 2.0]))
    u_half2 = surrogate.predict(u_half, params, np.array([t + dt]))
    return float(np.linalg.norm(u_full - u_half2) / (np.linalg.norm(u_t) + 1e-12))


# ---------------------------------------------------------------------------
# 3. Physical constraint deviation (Burgers: mass conservation)
# ---------------------------------------------------------------------------

def mass_conservation_deviation(u_t: np.ndarray, u_pred: np.ndarray) -> float:
    """|mean(u_pred) - mean(u_t)|; periodic Burgers conserves mass."""
    return float(abs(np.mean(u_pred) - np.mean(u_t)) / (abs(np.mean(u_t)) + 1e-12))


# ---------------------------------------------------------------------------
# 4. Increment magnitude
# ---------------------------------------------------------------------------

def increment_magnitude(u_t: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_pred - u_t) / (np.linalg.norm(u_t) + 1e-12))


# ---------------------------------------------------------------------------
# 5. Local gradient magnitude
# ---------------------------------------------------------------------------

def gradient_magnitude(u_t: np.ndarray, L: float = 1.0) -> float:
    dx = _dx_from_state(u_t, L)
    ux = periodic_gradient(u_t, dx)
    return float(np.linalg.norm(ux) / (np.linalg.norm(u_t) + 1e-12))


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

def extract_features(
    surrogate,
    u_t: np.ndarray,
    u_pred: np.ndarray,
    params: np.ndarray,      # e.g. [nu]
    t: float,
    dt: float,
    L: float = 1.0,
) -> np.ndarray:
    """Return a fixed-length risk feature vector (no future ground truth)."""
    nu = float(params[0]) if params.size else 0.01
    f_res = normalized_residual(u_t, u_pred, nu, dt, L)
    f_temp = temporal_inconsistency(surrogate, u_t, params, t, dt)
    f_mass = mass_conservation_deviation(u_t, u_pred)
    f_inc = increment_magnitude(u_t, u_pred)
    f_grad = gradient_magnitude(u_t, L)
    # params themselves are informative (higher nu -> smoother -> easier)
    return np.array([f_res, f_temp, f_mass, f_inc, f_grad, nu], dtype=np.float64)


def feature_names() -> list[str]:
    return ["norm_residual", "temporal_inconsistency", "mass_deviation",
            "increment_magnitude", "gradient_magnitude", "nu"]
