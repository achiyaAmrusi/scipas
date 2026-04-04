import numpy as np
import pytest
import xarray as xr

from pyPAS.core.lt import PASLifetime, MeasuredRF, MultiGaussianRF
from pyPAS.lifetime.model import LifetimeModel
from pyPAS.lifetime.generator import generate_lt_spectrum
from pyspectrum.spectrum import Spectrum
# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_lt_spectrum():
    time = np.arange(-2, 10, 0.05)
    channels = range(len(time))

    counts = np.zeros_like(channels)
    resolution_function = MultiGaussianRF(sigmas=np.array([.25]), weights=np.array([1]), t0=np.array([0]))
    model = LifetimeModel(name='test', lifetimes=[1], intensities=[1] )
    counts = generate_lt_spectrum(time=time, model=model, resolution= resolution_function)
    counts += 10  # flat background
    return Spectrum(
        counts=counts,
        channels=channels
    )

