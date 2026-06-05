"""
Validates profile_solver against the closed-form analytical solution for a
homogeneous medium with a uniform (constant) source and a uniform electric field.

PDE:   D c'' - v c' - λ c = -I₀        (v = μ·E, constant throughout)
BCs:   c'(0) =  c(0) / L_a             (surface, radiative)
       c'(L) = -c(L) / L_diff          (back surface, radiative;  L_diff = sqrt(D/λ))

Exact solution (stable form):
    c(z) = Ã·exp(κ₁(z-L)) + B·exp(κ₂z) + I₀/λ

    κ₁ = v/(2D) + sqrt((v/2D)² + 1/L_diff²)      (positive)
    κ₂ = v/(2D) - sqrt((v/2D)² + 1/L_diff²)      (negative for small v)

    Ã, B solved from the 2×2 linear system enforcing both BCs.
    The Ã·exp(κ₁(z-L)) form avoids exp(κ₁·L)≫1 overflow for large L or κ₁.
"""

import numpy as np
import xarray as xr
import pytest

from pyPAS.model.material import Material
from pyPAS.model.layer import Layer
from pyPAS.model.sample import Sample
from pyPAS.transport.diffusion.positron_profile_solver import profile_solver


# ── analytical reference ──────────────────────────────────────────────────────

def _analytical(z: np.ndarray, D: float, v: float, lam: float,
                La: float, L: float, I0: float = 1.0) -> np.ndarray:
    """Exact solution for the constant-source BVP on [0, L]."""
    L_diff = np.sqrt(D / lam)
    half_v = v / (2.0 * D)
    disc = np.sqrt(half_v**2 + 1.0 / L_diff**2)
    kappa_1 = half_v + disc   # > 0
    kappa_2 = half_v - disc   # < 0 (for small v)

    # Stable 2×2 system:  avoids exp(kappa_1*L) overflow
    bc_mat = np.array([
        [np.exp(-kappa_1 * L) * (kappa_1 - 1.0 / La),
         kappa_2 - 1.0 / La],
        [kappa_1 + 1.0 / L_diff,
         np.exp(kappa_2 * L) * (kappa_2 + 1.0 / L_diff)],
    ])
    rhs = np.array([(I0 / lam) / La, -(I0 / lam) / L_diff])
    A_tilde, B = np.linalg.solve(bc_mat, rhs)

    return A_tilde * np.exp(kappa_1 * (z - L)) + B * np.exp(kappa_2 * z) + I0 / lam


# ── parametrized test ─────────────────────────────────────────────────────────

# Each tuple: (label, D, mobility, E_val, lam, La, sample_len)
# v = mobility * E_val; L_diff = sqrt(D/lam)
_CASES = [
    # no drift — baseline
    ("no_drift",          1.0, 0.0,  0.0, 1e-4, 2.0, 400.0),
    # moderate drift  (v = 0.1·D/L_diff)
    ("drift_moderate",    1.0, 0.1,  1.0, 1e-4, 2.0, 400.0),
    # stronger drift  (v = D/L_diff)
    ("drift_strong",      1.0, 1.0,  1.0, 1e-4, 2.0, 400.0),
    # shorter diffusion length, small surface absorption
    ("short_Ldiff",       1.0, 0.0,  0.0, 4e-4, 0.5, 400.0),
    # larger La (less surface trapping)
    ("large_La_drift",    1.0, 0.5,  1.0, 1e-4, 10.0, 400.0),
]


@pytest.mark.parametrize("label,D,mobility,E_val,lam,La,sample_len", _CASES,
                         ids=[c[0] for c in _CASES])
def test_constant_source_analytical(label, D, mobility, E_val, lam, La, sample_len):
    """
    FD solver matches the analytical constant-source solution to better than 0.01 %.

    Error metric: max_z |c_fd(z) - c_exact(z)| / max_z c_exact(z)

    Mesh: 8000 cells over sample_len = 400 nm  →  h = 0.05 nm.
    From the O(h²) convergence measured in the benchmark notebook, this gives
    ~0.003 % max normalised error — well within the 0.01 % threshold.
    """
    mesh_size = 8000
    I0 = 1.0
    v = mobility * E_val

    # constant source
    n_src = 2000
    x_src = np.linspace(0, sample_len, n_src)
    source = xr.DataArray(np.full(n_src, I0), coords={"x": x_src})

    mat = Material(name="bulk", diffusion=D, mobility=mobility,
                   bulk_annihilation_rate=lam)
    sample = Sample(layers=[Layer(material=mat, width=sample_len)],
                    absorption_length=La)

    if v != 0.0:
        E_field = xr.DataArray(np.full(n_src, E_val), coords={"x": x_src})
        fd = profile_solver(source, sample, electric_field=E_field, mesh_size=mesh_size)
    else:
        fd = profile_solver(source, sample, mesh_size=mesh_size)

    z = fd.coords["x"].values
    c_exact = _analytical(z, D, v, lam, La, sample_len, I0)
    c_fd = fd.values

    rel_error = np.abs(c_fd - c_exact) / c_exact.max()
    max_err = rel_error.max()

    assert max_err < 1e-4, (
        f"[{label}] max relative error {max_err:.2e} exceeds 0.01 %"
    )
