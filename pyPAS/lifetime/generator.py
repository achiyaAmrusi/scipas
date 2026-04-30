import numpy as np
from pyPAS.lifetime.model import LifetimeModel
from pyPAS.core.lt import TimeResolution, PASLifetime
from pyspectrum import Spectrum

def generate_analytical_lt_spectrum(
    time: np.ndarray,
    model: LifetimeModel,
    resolution: TimeResolution) -> PASLifetime:
    """
    Generate a normalized positron fit spectrum on given time grid
    using discrete exponential model and resolution convolution.
    """

    # ---- Validate time axis ----
    if time.ndim != 1:
        raise ValueError("Time must be 1D array")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis must be strictly increasing")

    # ---- Generate analytical decay ----
    decay = np.zeros_like(time, dtype=float)
    lifetime = np.vstack(model.lifetimes)
    intensity = np.vstack(model.intensities)

    index_time_0 = np.where(time>0)[0][0]
    decay[index_time_0:] = (intensity / lifetime * np.exp(-time[index_time_0:] / lifetime)).sum(axis=0)
    decay = resolution.convolve(decay, time)

    #----- Normalize the Decay to Background ---
    decay = decay / np.trapz(decay, time)

    # ---- Time calibration (linear) ----
    dt = time[1] - time[0]
    time_calib = np.poly1d([dt, time[0]])

    spectrum = Spectrum(
        counts=decay,
        channels=np.arange(len(time)),
        energy_calibration_poly=time_calib
    )
    return PASLifetime(lifetime=spectrum, resolution=resolution)

def generate_random_lt_spectrum(
        time: np.ndarray,
        model: LifetimeModel,
        resolution: TimeResolution,
        num_events = int
) -> PASLifetime:
    """
    Generate a normalized positron fit spectrum on given time grid
    using discrete exponential model and resolution convolution.
    """
    dt = time[1] - time[0]
    analytical = generate_analytical_lt_spectrum(time=time,
                                                 model=model,
                                                 resolution=resolution)
    measured = np.random.poisson(analytical.lifetime * dt * num_events).astype(float)

    # ---- Time calibration (linear) ----
    dt = time[1] - time[0]
    time_calib = np.poly1d([dt, time[0]])

    spectrum = Spectrum(
        counts=measured,
        channels=np.arange(len(time)),
        energy_calibration_poly=time_calib
    )

    return PASLifetime(lifetime=spectrum, resolution=resolution)
