import numpy as np
import pandas as pd
import pytest
import xarray as xr
from uncertainties import ufloat

from scipas.model import Material, Layer, Sample
from scipas.analysis.vedb.diffusion_length import DiffusionLengthOptimization

# ── helpers ───────────────────────────────────────────────────────────────────

MESH = 500   # small mesh so tests run fast


def _gaussian_profile(center: float, depth: np.ndarray, sigma: float = 100.0) -> xr.DataArray:
    p = np.exp(-0.5 * ((depth - center) / sigma) ** 2)
    return xr.DataArray(p / np.trapezoid(p, depth), coords={'x': depth})


def _single_layer_sample(L: float, width: float = 5000.0) -> Sample:
    mat = Material(diffusion=1.0, mobility=0.0, bulk_annihilation_rate=1.0 / L ** 2)
    return Sample(layers=[Layer(width=width, material=mat)], absorption_length=1.0)


def _two_layer_sample(L0: float, L1: float,
                      w0: float = 500.0, w1: float = 5000.0) -> Sample:
    m0 = Material(diffusion=1.0, mobility=0.0, bulk_annihilation_rate=1.0 / L0 ** 2)
    m1 = Material(diffusion=1.0, mobility=0.0, bulk_annihilation_rate=1.0 / L1 ** 2)
    return Sample(layers=[Layer(width=w0, material=m0),
                          Layer(width=w1, material=m1)], absorption_length=1.0)


def _synthetic_s_measurement(optimizer: DiffusionLengthOptimization,
                              true_sample: Sample,
                              s_per_channel: np.ndarray,
                              sigma: float = 5e-4) -> pd.Series:
    """Run the forward model on true_sample and return noisy ufloat S measurements."""
    frac = optimizer.layers_transport_solver(true_sample, optimizer.positron_implantation_profiles)
    s_vals = frac @ s_per_channel
    return pd.Series([ufloat(float(s), sigma) for s in s_vals])


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def depth():
    return np.arange(0.0, 5000.0, 5.0)


@pytest.fixture
def single_layer_profiles(depth):
    centers = np.linspace(50, 2000, 6)
    return [_gaussian_profile(c, depth) for c in centers]


@pytest.fixture
def two_layer_profiles(depth):
    centers = np.linspace(50, 3000, 8)
    return [_gaussian_profile(c, depth) for c in centers]


@pytest.fixture
def single_layer_optimizer(single_layer_profiles):
    """Optimizer pre-loaded with synthetic S data from a true sample (L=100 nm)."""
    true_L = 100.0
    true_sample = _single_layer_sample(true_L)
    init_sample = _single_layer_sample(L=150.0)   # deliberately wrong initial guess

    # bootstrap: build optimizer once just to run the forward model
    bootstrap = DiffusionLengthOptimization(
        single_layer_profiles,
        pd.Series([ufloat(0.5, 1e-3)] * len(single_layer_profiles)),
        true_sample, num_of_mesh_cells=MESH)
    s_meas = _synthetic_s_measurement(bootstrap, true_sample,
                                      s_per_channel=np.array([0.45, 0.52]))

    return DiffusionLengthOptimization(
        single_layer_profiles, s_meas, init_sample, num_of_mesh_cells=MESH)


@pytest.fixture
def two_layer_optimizer(two_layer_profiles):
    """Optimizer for a two-layer sample (L0=80 nm, L1=150 nm)."""
    true_L0, true_L1 = 80.0, 150.0
    true_sample = _two_layer_sample(true_L0, true_L1)
    init_sample = _two_layer_sample(L0=120.0, L1=200.0)

    bootstrap = DiffusionLengthOptimization(
        two_layer_profiles,
        pd.Series([ufloat(0.5, 1e-3)] * len(two_layer_profiles)),
        true_sample, num_of_mesh_cells=MESH)
    s_meas = _synthetic_s_measurement(bootstrap, true_sample,
                                      s_per_channel=np.array([0.44, 0.50, 0.53]))

    return DiffusionLengthOptimization(
        two_layer_profiles, s_meas, init_sample, num_of_mesh_cells=MESH)


# ── make_sample ───────────────────────────────────────────────────────────────

def test_make_sample_layer_count(single_layer_optimizer):
    s = single_layer_optimizer.make_sample([100.0])
    assert len(s.layers) == 1


def test_make_sample_diffusion_length(single_layer_optimizer):
    L = 123.0
    s = single_layer_optimizer.make_sample([L])
    mat = s.layers[0].material
    assert np.isclose(np.sqrt(mat.diffusion / mat.bulk_annihilation_rate), L)


def test_make_sample_preserves_geometry(single_layer_optimizer):
    s = single_layer_optimizer.make_sample([100.0])
    ref = single_layer_optimizer.initial_sample
    for i, layer in enumerate(s.layers):
        assert np.isclose(layer.width, ref.layers[i].width)


# ── layers_transport_solver ───────────────────────────────────────────────────

def test_transport_solver_matrix_shape(single_layer_optimizer):
    sample = single_layer_optimizer.make_sample([100.0])
    frac = single_layer_optimizer.layers_transport_solver(
        sample, single_layer_optimizer.positron_implantation_profiles)
    n_profiles = len(single_layer_optimizer.positron_implantation_profiles)
    assert frac.shape == (n_profiles, single_layer_optimizer.n_layers + 1)


def test_transport_solver_rows_sum_to_one(single_layer_optimizer):
    sample = single_layer_optimizer.make_sample([100.0])
    frac = single_layer_optimizer.layers_transport_solver(
        sample, single_layer_optimizer.positron_implantation_profiles)
    assert np.allclose(frac.sum(axis=1), 1.0, atol=1e-3)


def test_transport_solver_nonnegative(single_layer_optimizer):
    sample = single_layer_optimizer.make_sample([100.0])
    frac = single_layer_optimizer.layers_transport_solver(
        sample, single_layer_optimizer.positron_implantation_profiles)
    assert np.all(frac >= 0)


# ── layer_s_value ─────────────────────────────────────────────────────────────

def test_layer_s_value_shape(single_layer_optimizer):
    sample = single_layer_optimizer.make_sample([100.0])
    frac = single_layer_optimizer.layers_transport_solver(
        sample, single_layer_optimizer.positron_implantation_profiles)
    s_vec = single_layer_optimizer.layer_s_value(frac)
    assert s_vec.shape == (single_layer_optimizer.n_layers + 1,)


# ── residuals ─────────────────────────────────────────────────────────────────

def test_residuals_shape(single_layer_optimizer):
    n_profiles = len(single_layer_optimizer.positron_implantation_profiles)
    r = single_layer_optimizer.residuals(np.array([100.0]))
    assert r.shape == (n_profiles,)


def test_residuals_near_zero_at_truth(single_layer_optimizer):
    """At the true diffusion length the residuals should be small."""
    r = single_layer_optimizer.residuals(np.array([100.0]))
    assert np.all(np.abs(r) < 5.0)


# ── optimize_diffusion_length — output contract ───────────────────────────────

def test_optimize_returns_tuple(single_layer_optimizer):
    result = single_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    assert isinstance(result, tuple) and len(result) == 2


def test_optimize_best_fit_shape(single_layer_optimizer):
    best_fit, _ = single_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    assert best_fit.shape == (single_layer_optimizer.n_layers,)


def test_optimize_covariance_shape(single_layer_optimizer):
    _, cov = single_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    n = single_layer_optimizer.n_layers
    assert cov.shape == (n, n)


def test_optimize_covariance_symmetric(single_layer_optimizer):
    _, cov = single_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    assert np.allclose(cov, cov.T)


def test_optimize_covariance_positive_variances(single_layer_optimizer):
    _, cov = single_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    assert np.all(np.diag(cov) >= 0)


def test_optimize_invalid_bounds_raises(single_layer_optimizer):
    with pytest.raises(ValueError):
        single_layer_optimizer.optimize_diffusion_length(bounds=(500, 1))


# ── single-layer diffusion length recovery ────────────────────────────────────

def test_single_layer_recovery(single_layer_optimizer):
    """Optimizer must recover L=100 nm from noise-free synthetic data."""
    best_fit, _ = single_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    assert np.isclose(best_fit[0], 100.0, rtol=0.10), \
        f"Expected ~100 nm, got {best_fit[0]:.1f} nm"


# ── two-layer diffusion length recovery ───────────────────────────────────────

def test_two_layer_recovery(two_layer_optimizer):
    """Optimizer must recover L0=80 nm, L1=150 nm from noise-free synthetic data."""
    best_fit, cov = two_layer_optimizer.optimize_diffusion_length(bounds=(1, 500))
    assert best_fit.shape == (2,)
    assert np.isclose(best_fit[0], 80.0, rtol=0.15), \
        f"Layer 0: expected ~80 nm, got {best_fit[0]:.1f} nm"
    assert np.isclose(best_fit[1], 150.0, rtol=0.15), \
        f"Layer 1: expected ~150 nm, got {best_fit[1]:.1f} nm"
    assert cov.shape == (2, 2)
