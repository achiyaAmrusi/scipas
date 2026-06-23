import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.inversion.tikhonov import TikhonovRegularization
from scipas.analysis.lifetime.inversion.maximum_entropy import MaximalEntropyInversion
from scipas.core.lifetime import PASLifetime
from scispectrum import Spectrum


@pytest.fixture
def synthetic_pals():
    np.random.seed(42)
    time = np.arange(-2, 15, 0.01)
    sigma = np.array([0.230 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))

    tau_grid_fine = np.arange(0.005, 5, 0.005)
    intensities = np.zeros_like(tau_grid_fine)
    for tau_c, w, sig in [(0.4, 0.7, 0.02), (2.0, 0.3, 0.1)]:
        intensities += w * np.exp(-((tau_grid_fine - tau_c) ** 2) / (2 * sig ** 2))
    intensities /= np.trapezoid(intensities, tau_grid_fine)

    model = LifetimeModel("test", lifetimes=tau_grid_fine, intensities=intensities)
    bg = 50.0
    r = generate_random_lt_spectrum(time, model, irf, num_events=1_000_000)
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf,
    )
    tau_grid = np.arange(0.05, 5, 0.05)
    return pals, tau_grid, bg


def test_tikhonov_produces_distribution(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    tik = TikhonovRegularization(pals.lifetime.energy.values, tau_grid)
    q, res = tik.invert(pals, bg_est=bg, initial_alpha=1e-5)
    assert q.shape == tau_grid.shape
    assert np.all(q >= 0)
    assert res.success


def test_tikhonov_distribution_integrates_near_one(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    tik = TikhonovRegularization(pals.lifetime.energy.values, tau_grid)
    q, _ = tik.invert(pals, bg_est=bg)
    integral = np.trapezoid(q, tau_grid)
    assert abs(integral - 1.0) < 0.1


def test_tikhonov_peak_location(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    tik = TikhonovRegularization(pals.lifetime.energy.values, tau_grid)
    q, _ = tik.invert(pals, bg_est=bg)
    peak_tau = tau_grid[np.argmax(q)]
    assert 0.2 < peak_tau < 0.8


def test_tikhonov_t0_shift(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    tik = TikhonovRegularization(pals.lifetime.energy.values, tau_grid)
    q0, _ = tik.invert(pals, bg_est=bg, t0_shift=0.0)
    q1, _ = tik.invert(pals, bg_est=bg, t0_shift=0.005)
    assert not np.allclose(q0, q1)


def test_melt_produces_distribution(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    melt = MaximalEntropyInversion(pals.lifetime.energy.values, tau_grid)
    alpha, f = melt.invert(pals, bg_est=bg, noise_level=1e-3, maxiter=100)
    assert f.shape == tau_grid.shape
    assert np.all(f >= 0)
    assert alpha > 0


def test_melt_peak_location(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    melt = MaximalEntropyInversion(pals.lifetime.energy.values, tau_grid)
    _, f = melt.invert(pals, bg_est=bg, noise_level=1e-3, maxiter=100)
    peak_tau = tau_grid[np.argmax(f)]
    assert 0.2 < peak_tau < 0.8


def test_melt_t0_shift(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    melt = MaximalEntropyInversion(pals.lifetime.energy.values, tau_grid)
    _, f0 = melt.invert(pals, bg_est=bg, t0_shift=0.0, maxiter=50)
    _, f1 = melt.invert(pals, bg_est=bg, t0_shift=0.005, maxiter=50)
    assert not np.allclose(f0, f1)


def test_imports_from_public_api():
    from scipas.analysis.lifetime import (
        TikhonovRegularization,
        MaximalEntropyInversion,
        LifetimeInvert,
    )
    assert issubclass(TikhonovRegularization, LifetimeInvert)
    assert issubclass(MaximalEntropyInversion, LifetimeInvert)
