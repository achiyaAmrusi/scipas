import numpy as np
import pytest
from pypas.core.time_resolution import MultiGaussianRF
from pypas.model.lifetime import LifetimeModel
from pypas.analysis.lifetime.generator import (
    generate_analytical_lt_spectrum,
    generate_random_lt_spectrum,
)
from pypas.core.lifetime import PASLifetime


@pytest.fixture
def setup():
    time = np.arange(-2, 10, 0.01)
    sigma = np.array([0.200 / (2 * np.sqrt(2 * np.log(2)))])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("test", lifetimes=[0.4, 2.0], intensities=[0.7, 0.3])
    return time, irf, model


def test_analytical_returns_pas_lifetime(setup):
    time, irf, model = setup
    result = generate_analytical_lt_spectrum(time, model, irf)
    assert isinstance(result, PASLifetime)
    assert result.lifetime.counts.shape == time.shape


def test_analytical_normalized_to_one(setup):
    time, irf, model = setup
    result = generate_analytical_lt_spectrum(time, model, irf)
    integral = np.trapezoid(result.lifetime.counts, time)
    assert abs(integral - 1.0) < 1e-6


def test_analytical_non_negative(setup):
    time, irf, model = setup
    result = generate_analytical_lt_spectrum(time, model, irf)
    assert np.all(result.lifetime.counts >= -1e-15)


def test_analytical_peak_after_t0(setup):
    time, irf, model = setup
    result = generate_analytical_lt_spectrum(time, model, irf)
    peak_idx = np.argmax(result.lifetime.counts)
    peak_time = time[peak_idx]
    assert peak_time >= -0.5


def test_random_total_counts(setup):
    time, irf, model = setup
    np.random.seed(123)
    num_events = 500_000
    result = generate_random_lt_spectrum(time, model, irf, num_events=num_events)
    total = result.lifetime.counts.sum()
    assert abs(total - num_events) / num_events < 0.02


def test_random_is_poisson(setup):
    time, irf, model = setup
    np.random.seed(456)
    result = generate_random_lt_spectrum(time, model, irf, num_events=1_000_000)
    counts = result.lifetime.counts
    assert np.all(counts >= 0)
    assert counts.dtype == np.float64
    assert np.all(counts == np.floor(counts))


def test_analytical_time_axis_matches(setup):
    time, irf, model = setup
    result = generate_analytical_lt_spectrum(time, model, irf)
    energy_vals = result.lifetime.energy.values
    assert np.allclose(time, energy_vals)


def test_single_component(setup):
    time, irf, _ = setup
    model = LifetimeModel("single", lifetimes=[1.0], intensities=[1.0])
    result = generate_analytical_lt_spectrum(time, model, irf)
    integral = np.trapezoid(result.lifetime.counts, time)
    assert abs(integral - 1.0) < 1e-6


def test_invalid_time_raises():
    time_2d = np.ones((3, 3))
    sigma = np.array([0.1])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("test", lifetimes=[1.0], intensities=[1.0])
    with pytest.raises(ValueError, match="1D"):
        generate_analytical_lt_spectrum(time_2d, model, irf)


def test_non_monotonic_time_raises():
    time = np.array([0, 1, 0.5, 2])
    sigma = np.array([0.1])
    irf = MultiGaussianRF(sigma, np.ones_like(sigma), np.zeros_like(sigma))
    model = LifetimeModel("test", lifetimes=[1.0], intensities=[1.0])
    with pytest.raises(ValueError, match="increasing"):
        generate_analytical_lt_spectrum(time, model, irf)
