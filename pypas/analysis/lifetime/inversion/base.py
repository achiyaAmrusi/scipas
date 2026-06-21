from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
from pypas.core.lifetime import PASLifetime


class LifetimeInvert(ABC):
    def __init__(self, time_grid: np.ndarray, characteristic_time_grid: np.ndarray):
        if time_grid.ndim != 1 or characteristic_time_grid.ndim != 1:
            raise ValueError("time and tau must be 1D arrays")
        self.time_grid = time_grid
        self.characteristic_time_grid = characteristic_time_grid

    @abstractmethod
    def invert(self, pals: PASLifetime, **kwargs) -> xr.DataArray:
        """Invert the decaying positron lifetime into its components."""
        pass