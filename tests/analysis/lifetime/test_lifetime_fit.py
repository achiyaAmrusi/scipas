import numpy as np
import pytest
from scipas.core.time_resolution import MultiGaussianRF
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_random_lt_spectrum
from scipas.analysis.lifetime.fit import LifetimeFitter, FitParameter, FitResult
from scipas.core.lifetime import PASLifetime
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


def _shifted_spectrum(t0_true, seed=7, num_events=1_000_000):
    """Two-component spectrum with a genuine time-zero offset.

    The offset is injected by centering the generating IRF at ``t0_true``;
    the fit is then given a *centered* IRF so the free t0 parameter must
    absorb the shift. This is an independent check of t0 recovery (the
    generator never uses the fitter's forward model).
    """
    np.random.seed(seed)
    time = np.arange(-2, 15, 0.01)
    sigma = np.array([0.200 / (2 * np.sqrt(2 * np.log(2)))])
    irf_shifted = MultiGaussianRF(sigma, np.ones_like(sigma), np.array([t0_true]))
    irf_centered = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("true", lifetimes=[0.2, 1.5], intensities=[0.6, 0.4])
    bg = 50.0
    r = generate_random_lt_spectrum(time, model, irf_shifted, num_events=num_events)
    pals = PASLifetime(
        lifetime=Spectrum(counts=r.lifetime.counts + bg, axis_calib=r.lifetime.axis_calib),
        resolution=irf_centered,
    )
    return pals, bg


@pytest.mark.parametrize("t0_true", [-0.05, 0.08, 0.15])
def test_fit_recovers_t0(t0_true):
    pals, bg = _shifted_spectrum(t0_true)
    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.15), FitParameter(1.2)],
        intensities=[FitParameter(0.6), FitParameter(0.4)],
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(bg, lower=0.0),
    )
    assert result.success
    # t0 recovered to well under the bin spacing (0.01 ns)
    assert abs(result.t0 - t0_true) < 0.01
    # lifetimes must not be corrupted by the shift
    taus = sorted(result.model.lifetimes)
    assert abs(taus[0] - 0.2) < 0.03
    assert abs(taus[1] - 1.5) < 0.15
    # a good fit at the recovered t0
    assert result.reduced_chi_squared < 2.0


def test_free_t0_has_finite_error():
    """Regression: the t0 Jacobian column must be non-degenerate.

    With the old support-truncation forward model the t0 finite-difference
    column was ~0, giving an absurd (1e5) error and a stalled fit.
    """
    pals, bg = _shifted_spectrum(0.08)
    fitter = LifetimeFitter()
    result = fitter.fit(
        pals,
        lifetimes=[FitParameter(0.15), FitParameter(1.2)],
        intensities=[FitParameter(0.6), FitParameter(0.4)],
        t0=FitParameter(0.0, lower=-0.5, upper=0.5),
        background=FitParameter(bg, lower=0.0),
    )
    assert np.isfinite(result.parameter_errors["t0"])
    assert result.parameter_errors["t0"] < 0.05


def test_fixed_nonzero_t0_shifts_model(two_component_spectrum):
    """A fixed non-zero t0 must move the model peak by that amount."""
    pals, _, bg = two_component_spectrum
    fitter = LifetimeFitter()
    args = dict(
        lifetimes=[FitParameter(0.2, fixed=True), FitParameter(1.5, fixed=True)],
        intensities=[FitParameter(0.6, fixed=True), FitParameter(0.4, fixed=True)],
        background=FitParameter(bg, lower=0.0),
    )
    base = fitter.fit(pals, t0=FitParameter(0.0, fixed=True), **args)
    shifted = fitter.fit(pals, t0=FitParameter(0.10, fixed=True), **args)
    time = pals.lifetime.energy.values
    peak_base = time[np.argmax(base.fitted_spectrum)]
    peak_shifted = time[np.argmax(shifted.fitted_spectrum)]
    assert peak_shifted - peak_base == pytest.approx(0.10, abs=0.02)
