import numpy as np
from pyPAS.model.lifetime import LifetimeModel
from pyPAS.core.lifetime import TimeResolution, PASLifetime
from scispectrum import Spectrum
from scispectrum.calibration.axis import AxisCalibration


def _time_axis_calibration(time: np.ndarray) -> AxisCalibration:
    dt = time[1] - time[0]
    t0 = time[0]
    return AxisCalibration(lambda ch, _dt=dt, _t0=t0: ch * _dt + _t0, name="energy")


def generate_analytical_lt_spectrum(
    time: np.ndarray,
    model: LifetimeModel,
    resolution: TimeResolution) -> PASLifetime:
    """
    Generate a normalized positron lifetime spectrum on a given time grid
    using a discrete exponential model convolved with the resolution function.
    """

    if time.ndim != 1:
        raise ValueError("Time must be 1D array")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis must be strictly increasing")

    decay = np.zeros_like(time, dtype=float)
    lifetime = np.vstack(model.lifetimes)
    intensity = np.vstack(model.intensities)

    index_time_0 = np.where(time > 0)[0][0]
    decay[index_time_0:] = (intensity / lifetime * np.exp(-time[index_time_0:] / lifetime)).sum(axis=0)
    decay = resolution.convolve(decay, time)

    decay = decay / np.trapezoid(decay, time)

    spectrum = Spectrum(
        counts=decay,
        axis_calib=_time_axis_calibration(time),
    )
    return PASLifetime(lifetime=spectrum, resolution=resolution)


def generate_random_lt_spectrum(
        time: np.ndarray,
        model: LifetimeModel,
        resolution: TimeResolution,
        num_events: int = 1_000_000,
) -> PASLifetime:
    """
    Generate a Poisson-sampled positron lifetime spectrum on a given time grid
    using a discrete exponential model convolved with the resolution function.
    """
    dt = time[1] - time[0]
    analytical = generate_analytical_lt_spectrum(time=time,
                                                 model=model,
                                                 resolution=resolution)
    measured = np.random.poisson(analytical.lifetime.counts * dt * num_events).astype(float)

    spectrum = Spectrum(
        counts=measured,
        axis_calib=_time_axis_calibration(time),
    )

    return PASLifetime(lifetime=spectrum, resolution=resolution)
