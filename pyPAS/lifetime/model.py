from dataclasses import dataclass, field
import numpy as np
from typing import Sequence


@dataclass
class LifetimeModel:
    """
    Represents a discrete multi-component fit materials.

    lifetimes: decay constants (tau_i)
    intensities: relative intensities (I_i)
    """

    name: str
    lifetimes: Sequence[float] | np.ndarray = field(default_factory=list)
    intensities: Sequence[float] | np.ndarray = field(default_factory=list)

    def __post_init__(self):

        # Convert to numpy arrays
        self.lifetimes = np.asarray(self.lifetimes, dtype=float)
        self.intensities = np.asarray(self.intensities, dtype=float)

        # ---- Validation ----

        if self.lifetimes.ndim != 1:
            raise ValueError("lifetimes must be a 1D sequence")

        if self.intensities.ndim != 1:
            raise ValueError("intensities must be a 1D sequence")

        if len(self.lifetimes) == 0:
            raise ValueError("At least one fit component is required")

        if len(self.lifetimes) != len(self.intensities):
            raise ValueError(
                "Number of lifetimes must match number of intensities"
            )

        if np.any(self.lifetimes <= 0):
            raise ValueError("All lifetimes must be positive")

        if np.any(self.intensities < 0):
            raise ValueError("Intensities must be non-negative")

        total_intensity = self.intensities.sum()

        if total_intensity == 0:
            raise ValueError("Sum of intensities must be > 0")

        # ---- Normalize intensities ----
        self.intensities = self.intensities / total_intensity
