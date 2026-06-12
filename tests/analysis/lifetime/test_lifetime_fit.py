import numpy as np
import pytest
from pyPAS.core.time_resolution import MultiGaussianRF
from pyPAS.model.lifetime import LifetimeModel
from pyPAS.analysis.lifetime.generator import generate_random_lt_spectrum
from pyPAS.analysis.lifetime.fit import LifetimeFitter, FitParameter, FitResult
from pyPAS.core.lifetime import PASLifetime
from scispectrum import Spectrum


@pytest.fixture
def two_component_spectrum():
    np.random.seed(42)
    time = np.arange(-2, 15, 0.01)
    sigma = np.array([0.200 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("true", lifetimes=[0.2, 1.5], intensities=[0.6, 0.4])
    bg = 50.0
    r = generate_random_lt_spectrum(time, model, irf, num_events=1_000_000)
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf,
    )
    return pals, model, bg


def test_fit_recovers_lifetimes(two_component_spectrum):
    pals, true_model, bg = two_component_spectrum
    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.15), FitParameter(1.2)],
        intensities=[FitParameter(0.6), FitParameter(0.4)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    assert result.success
    taus = sorted(result.model.lifetimes)
    assert abs(taus[0] - 0.2) < 0.03
    assert abs(taus[1] - 1.5) < 0.15


def test_fit_recovers_background(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.15), FitParameter(1.2)],
        intensities=[FitParameter(0.6), FitParameter(0.4)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(40.0, lower=0.0),
    )
    assert abs(result.background - bg) < 10


def test_fit_returns_fit_result(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.15), FitParameter(1.2)],
        intensities=[FitParameter(0.6), FitParameter(0.4)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, fixed=True),
    )
    assert isinstance(result, FitResult)
    assert isinstance(result.model, LifetimeModel)
    assert result.fitted_spectrum.shape == pals.lifetime.counts.shape
    assert result.residuals.shape == pals.lifetime.counts.shape
    assert result.covariance.shape[0] == result.n_free


def test_fixed_params_stay_fixed(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    fixed_tau = 0.2
    fixed_I = 0.6
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(fixed_tau, fixed=True), FitParameter(1.2)],
        intensities=[FitParameter(fixed_I, fixed=True), FitParameter(0.4)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, lower=0.0),
    )
    assert result.model.lifetimes[0] == pytest.approx(fixed_tau, abs=1e-10)


def test_intensities_sum_to_one(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.15), FitParameter(1.2)],
        intensities=[FitParameter(0.6), FitParameter(0.4)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(bg, fixed=True),
    )
    assert abs(result.model.intensities.sum() - 1.0) < 1e-10


def test_single_component_fit():
    np.random.seed(99)
    time = np.arange(-2, 10, 0.01)
    sigma = np.array([0.15 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("single", lifetimes=[0.5], intensities=[1.0])
    r = generate_random_lt_spectrum(time, model, irf, num_events=500_000)
    bg = 20.0
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf,
    )

    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.4)],
        intensities=[FitParameter(1.0, fixed=True)],
        t0=FitParameter(0.0, fixed=True),
        background=FitParameter(15.0, lower=0.0),
    )
    assert result.success
    assert abs(result.model.lifetimes[0] - 0.5) < 0.05


def test_mismatched_lengths_raises():
    fitter = LifetimeFitter()
    pals = None  # won't get to use it
    with pytest.raises(ValueError, match="same length"):
        fitter.fit(
            pals,
            lifetimes=[FitParameter(0.5)],
            intensities=[FitParameter(0.5), FitParameter(0.5)],
        )


def test_no_free_params_raises(two_component_spectrum):
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    with pytest.raises(ValueError, match="No free"):
        fitter.fit(
            pals,
            lifetimes=[FitParameter(0.2, fixed=True)],
            intensities=[FitParameter(1.0, fixed=True)],
            t0=FitParameter(0.0, fixed=True),
            background=FitParameter(bg, fixed=True),
        )
