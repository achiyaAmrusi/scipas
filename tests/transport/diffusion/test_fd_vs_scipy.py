"""
Comparison tests between the finite-difference solver (profile_solver) and the
scipy BVP solver (scipy_profile_solver).

Both implement the steady-state positron diffusion-drift-annihilation BVP:

    d/dz[D(z) dc/dz] - μ(z)E(z) dc/dz - λ(z) c = -g(z)

with radiative boundary conditions:
    dc/dz|_{z=0}  =  c(0) / L_a          (surface absorption)
    dc/dz|_{z=L}  = -c(L) / L_bulk       (bulk diffusion tail)

Known limitations of the scipy solver
--------------------------------------
* Two-layer / discontinuous interfaces: solve_bvp expects a smooth ODE;
  discontinuous material coefficients at layer boundaries require excessive
  node refinement and the solver often exhausts max_nodes.  The corresponding
  test is marked xfail.
"""

import sys
import os
import numpy as np
import pytest
import xarray as xr

from pyPAS.model.material import Material
from pyPAS.model.layer import Layer
from pyPAS.model.sample import Sample
from pyPAS.transport.diffusion.positron_profile_solver import profile_solver

sys.path.insert(0, os.path.dirname(__file__))
from scipy_positron_profile_solver import scipy_profile_solver


# ── helpers ───────────────────────────────────────────────────────────────────

def _gaussian_source(length: float, center: float, sigma: float, n_pts: int = 800):
    x = np.linspace(0, length, n_pts)
    return xr.DataArray(np.exp(-0.5 * ((x - center) / sigma) ** 2), coords={"x": x})


def _uniform_field(length: float, E_val: float, n_pts: int = 500) -> xr.DataArray:
    x = np.linspace(0, length, n_pts)
    return xr.DataArray(np.full(n_pts, E_val), coords={"x": x})


def _normalized_l2(fd_result: xr.DataArray, scipy_sol, *, x_min=None, x_max=None) -> float:
    """Normalised RMS difference; optionally restricted to an interior window."""
    x  = fd_result.coords["x"].values
    fd = np.clip(fd_result.values,    0.0, None)
    sc = np.clip(scipy_sol.sol(x)[0], 0.0, None)
    if x_min is not None or x_max is not None:
        lo = x_min if x_min is not None else x.min()
        hi = x_max if x_max is not None else x.max()
        m  = (x >= lo) & (x <= hi)
        fd, sc = fd[m], sc[m]
    scale = max(fd.max(), sc.max())
    return 0.0 if scale == 0 else float(np.sqrt(np.mean(((fd - sc) / scale) ** 2)))


def _mean_depth(vals: np.ndarray, x: np.ndarray) -> float:
    v = np.clip(vals, 0.0, None)
    return float(np.trapezoid(x * v, x) / np.trapezoid(v, x))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def homogeneous_sample():
    """Single 300 nm layer. L+ = sqrt(D/λ) = 10 nm."""
    mat = Material(name="bulk", diffusion=0.1, mobility=0.0, bulk_annihilation_rate=0.001)
    return Sample(layers=[Layer(material=mat, width=300.0)], absorption_length=2.0)


@pytest.fixture
def drift_sample():
    """Same geometry as homogeneous_sample, non-zero mobility for drift tests."""
    mat = Material(name="bulk", diffusion=0.1, mobility=0.01, bulk_annihilation_rate=0.001)
    return Sample(layers=[Layer(material=mat, width=300.0)], absorption_length=2.0)


@pytest.fixture
def two_layer_sample():
    """20 nm high-annihilation surface layer on 280 nm bulk."""
    surface = Material(name="surface", diffusion=0.05, mobility=0.0, bulk_annihilation_rate=0.01)
    bulk    = Material(name="bulk",    diffusion=0.10, mobility=0.0, bulk_annihilation_rate=0.001)
    return Sample(
        layers=[Layer(material=surface, width=20.0), Layer(material=bulk, width=280.0)],
        absorption_length=2.0,
    )


# ── no-drift agreement tests ──────────────────────────────────────────────────

def test_single_layer_no_drift(homogeneous_sample):
    """FD and scipy agree to <2 % on a smooth single-layer, no-drift profile."""
    sample = homogeneous_sample
    source = _gaussian_source(sample.sample_length(), center=60.0, sigma=8.0)

    fd = profile_solver(source, sample, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"
    err = _normalized_l2(fd, sc)
    assert err < 0.02, f"normalised L2 error {err:.4f} exceeds 2 %"


def test_deep_source_no_drift(homogeneous_sample):
    """Both solvers agree when the source is far from both boundaries."""
    sample = homogeneous_sample
    source = _gaussian_source(sample.sample_length(), center=200.0, sigma=10.0)

    fd = profile_solver(source, sample, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"
    err = _normalized_l2(fd, sc)
    assert err < 0.02, f"normalised L2 error {err:.4f} exceeds 2 %"


def test_source_near_surface(homogeneous_sample):
    """Both solvers agree when the source is close to the surface (center = 30 nm = 3·L+).

    Note: sources within ~1–2·L+ of z=0 cause scipy's solve_bvp to fail because its
    initial guess (the raw implantation profile) does not satisfy the surface boundary
    condition, producing pathologically dense node spacing near z=0.  At 30 nm the
    initial-guess residual is small enough for the solver to converge.
    """
    sample = homogeneous_sample
    source = _gaussian_source(sample.sample_length(), center=30.0, sigma=6.0)

    fd = profile_solver(source, sample, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"
    assert np.all(fd.values >= -1e-8 * fd.values.max()), "FD profile has negative values"
    err = _normalized_l2(fd, sc)
    assert err < 0.02, f"normalised L2 error {err:.4f} exceeds 2 %"


def test_profile_peak_location_matches(homogeneous_sample):
    """Both solvers place the annihilation peak within two FD mesh steps."""
    sample = homogeneous_sample
    source = _gaussian_source(sample.sample_length(), center=60.0, sigma=8.0)

    fd = profile_solver(source, sample, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, num_of_mesh_cells=200)

    assert sc.success
    x  = fd.coords["x"].values
    dx = x[1] - x[0]
    fd_peak = x[np.argmax(fd.values)]
    sc_peak = x[np.argmax(sc.sol(x)[0])]
    assert abs(fd_peak - sc_peak) <= 2 * dx, (
        f"Peak locations differ by {abs(fd_peak - sc_peak):.3f} nm  (> 2·dx = {2*dx:.3f} nm)"
    )


def test_total_annihilation_budget(homogeneous_sample):
    """
    Both solvers report the same bulk-annihilation fraction to within 1 %.

    With a unit-normalised source (∫g dz = 1), ∫λ·c dz is the fraction of
    positrons that annihilate in the bulk (the rest escape at the boundaries).
    """
    sample = homogeneous_sample
    L      = sample.sample_length()
    x_src  = np.linspace(0, L, 2000)
    raw    = np.exp(-0.5 * ((x_src - 60.0) / 8.0) ** 2)
    raw   /= np.trapezoid(raw, x_src)
    source = xr.DataArray(raw, coords={"x": x_src})
    lam    = sample.layers[0].material.bulk_annihilation_rate

    fd = profile_solver(source, sample, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"

    x_fd      = fd.coords["x"].values
    budget_fd = float(np.trapezoid(lam * fd.values,            x_fd))
    budget_sc = float(np.trapezoid(lam * sc.sol(x_fd)[0], x_fd))

    assert 0 < budget_fd <= 1.05, f"FD budget = {budget_fd:.4f} outside (0, 1]"
    assert 0 < budget_sc <= 1.05, f"scipy budget = {budget_sc:.4f} outside (0, 1]"
    assert abs(budget_fd - budget_sc) < 0.01, (
        f"Annihilation budgets differ: FD={budget_fd:.4f}, scipy={budget_sc:.4f}"
    )


# ── drift agreement tests ─────────────────────────────────────────────────────

def test_drift_weak_field(drift_sample):
    """FD and scipy agree to <2 % under a weak uniform electric field (0.5 V/nm)."""
    sample = drift_sample
    source = _gaussian_source(sample.sample_length(), center=60.0, sigma=8.0)
    E      = _uniform_field(sample.sample_length(), 0.5)

    fd = profile_solver(source, sample, electric_field=E, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, electric_field=E, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"
    err = _normalized_l2(fd, sc, x_min=20.0, x_max=280.0)
    assert err < 0.02, f"interior normalised L2 error {err:.4f} exceeds 2 %"


def test_drift_strong_field(drift_sample):
    """FD and scipy agree to <2 % under a stronger field (2 V/nm)."""
    sample = drift_sample
    source = _gaussian_source(sample.sample_length(), center=60.0, sigma=8.0)
    E      = _uniform_field(sample.sample_length(), 2.0)

    fd = profile_solver(source, sample, electric_field=E, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, electric_field=E, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"
    err = _normalized_l2(fd, sc, x_min=20.0, x_max=280.0)
    assert err < 0.02, f"interior normalised L2 error {err:.4f} exceeds 2 %"


def test_drift_shifts_profile_deeper(drift_sample):
    """
    For E > 0 (field pointing into the material), positrons drift in +z.
    Both solvers should show an increased mean implantation depth, and
    reduced surface-region annihilation, relative to the no-drift case.
    """
    sample = drift_sample
    source = _gaussian_source(sample.sample_length(), center=60.0, sigma=8.0)
    E      = _uniform_field(sample.sample_length(), 1.0)

    fd_no = profile_solver(source, sample,                   mesh_size=3000)
    fd_dr = profile_solver(source, sample, electric_field=E,  mesh_size=3000)
    sc_no = scipy_profile_solver(source, sample,                   num_of_mesh_cells=200)
    sc_dr = scipy_profile_solver(source, sample, electric_field=E, num_of_mesh_cells=200)

    assert sc_no.success and sc_dr.success, "scipy BVP failed to converge"

    x = fd_no.coords["x"].values

    for label, no, dr in [("FD",    fd_no.values,     fd_dr.values),
                           ("scipy", sc_no.sol(x)[0],  sc_dr.sol(x)[0])]:
        shift = _mean_depth(dr, x) - _mean_depth(no, x)
        assert shift > 1.0, (
            f"{label}: mean depth should increase with E>0; shift = {shift:.2f} nm"
        )
        surf_no = float(np.trapezoid(no[x < 10], x[x < 10]))
        surf_dr = float(np.trapezoid(dr[x < 10], x[x < 10]))
        assert surf_dr < surf_no, (
            f"{label}: surface annihilation should decrease with drift away from surface"
        )


# ── known scipy limitation: two-layer convergence ────────────────────────────

@pytest.mark.xfail(
    strict=False,
    reason=(
        "scipy solve_bvp assumes a smooth ODE. Discontinuous material coefficients "
        "at layer interfaces require excessive node refinement; the solver often "
        "exhausts max_nodes without converging."
    ),
)
def test_two_layer_no_drift(two_layer_sample):
    """
    FD and scipy should agree on a two-layer profile if scipy converges.

    The FD solver handles material interfaces via interface-averaged diffusion
    coefficients, so it always converges.  The scipy solver struggles here; see
    the module docstring for details.
    """
    sample = two_layer_sample
    source = _gaussian_source(sample.sample_length(), center=50.0, sigma=8.0)

    fd = profile_solver(source, sample, mesh_size=3000)
    sc = scipy_profile_solver(source, sample, num_of_mesh_cells=200)

    assert sc.success, f"scipy BVP did not converge: {sc.message}"
    err = _normalized_l2(fd, sc)
    assert err < 0.03, f"normalised L2 error {err:.4f} exceeds 3 %"
