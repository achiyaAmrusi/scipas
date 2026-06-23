import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.inversion.gp_regression import GPRegression
from scipas.core.lifetime import PASLifetime
from scispectrum import Spectrum


@pytest.fixture
def synthetic_pals():
    np.random.seed(42)
    time = np.arange(-2, 15, 0.025)
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
    tau_grid = np.arange(0.1, 4, 0.1)
    return pals, tau_grid, bg


def test_gp_produces_distribution(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    gp = GPRegression(pals.lifetime.energy.values, tau_grid)
    f, meta = gp.invert(pals, bg_est=bg, optimize_hyperparams=False,
                        length_scale=0.5, log_amplitude=5.0)
    assert f.shape == tau_grid.shape
    assert np.all(f > 0)


def test_gp_returns_metadata(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    gp = GPRegression(pals.lifetime.energy.values, tau_grid)
    f, meta = gp.invert(pals, bg_est=bg, optimize_hyperparams=False,
                        length_scale=0.5, log_amplitude=5.0)
    assert 'posterior_std' in meta
    assert 'posterior_mean_log' in meta
    assert 'posterior_cov_log' in meta
    assert 'length_scale' in meta
    assert 'log_amplitude' in meta
    assert meta['posterior_std'].shape == tau_grid.shape
    assert meta['posterior_cov_log'].shape == (len(tau_grid), len(tau_grid))


def test_gp_posterior_std_positive(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    gp = GPRegression(pals.lifetime.energy.values, tau_grid)
    f, meta = gp.invert(pals, bg_est=bg, optimize_hyperparams=False,
                        length_scale=0.5, log_amplitude=5.0)
    assert np.all(meta['posterior_std'] >= 0)


def test_gp_peak_location(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    gp = GPRegression(pals.lifetime.energy.values, tau_grid)
    f, _ = gp.invert(pals, bg_est=bg, optimize_hyperparams=False,
                     length_scale=0.5, log_amplitude=5.0)
    peak_tau = tau_grid[np.argmax(f)]
    assert 0.2 < peak_tau < 0.8


def test_gp_with_hyperparameter_optimization(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    gp = GPRegression(pals.lifetime.energy.values, tau_grid)
    f, meta = gp.invert(pals, bg_est=bg, optimize_hyperparams=True)
    assert f.shape == tau_grid.shape
    assert np.all(f > 0)
    assert meta['length_scale'] > 0
    assert meta['log_amplitude'] > 0


def test_gp_t0_shift(synthetic_pals):
    pals, tau_grid, bg = synthetic_pals
    gp = GPRegression(pals.lifetime.energy.values, tau_grid)
    f0, _ = gp.invert(pals, bg_est=bg, t0_shift=0.0,
                      optimize_hyperparams=False,
                      length_scale=0.5, log_amplitude=5.0)
    f1, _ = gp.invert(pals, bg_est=bg, t0_shift=0.01,
                      optimize_hyperparams=False,
                      length_scale=0.5, log_amplitude=5.0)
    assert not np.allclose(f0, f1)


def test_gp_inherits_lifetime_invert():
    from scipas.analysis.lifetime.inversion import LifetimeInvert
    assert issubclass(GPRegression, LifetimeInvert)


def test_gp_import_from_public_api():
    from scipas.analysis.lifetime import GPRegression as GP
    assert GP is GPRegression
