import numpy as np
from dataclasses import dataclass


@dataclass
class FitParameter:
    """
    A single fittable parameter with optional bounds and fixed-value support.

    Parameters
    ----------
    value : float
        Initial guess (if free) or fixed value (if fixed).
    fixed : bool
        If True, parameter is held constant during fitting.
    lower : float
        Lower bound for optimization.
    upper : float
        Upper bound for optimization.
    """
    value: float
    fixed: bool = False
    lower: float = -np.inf
    upper: float = np.inf


class ParameterMap:
    """
    Maps between named fit parameters and the flat optimizer vector.

    Free parameters are ordered: lifetimes (tau_i), independent intensities
    (I_i), t0, background. Intensities are constrained to sum to 1: the last
    non-fixed intensity is the dependent one — computed as 1 - sum(others)
    in `unpack` and never part of the optimizer vector.

    Attributes
    ----------
    free_names : list of str
        Names of free parameters in optimizer-vector order.
    bounds_lower, bounds_upper : list of float
        Bounds for the free parameters, same order.
    dependent_intensity_idx : int or None
        Index of the intensity computed as remainder, or None if all
        intensities are fixed.
    """

    def __init__(self, lifetimes: list[FitParameter],
                 intensities: list[FitParameter],
                 t0: FitParameter, background: FitParameter):
        self._lifetimes = lifetimes
        self._intensities = intensities
        self._t0 = t0
        self._background = background

        n = len(lifetimes)
        non_fixed_I = [i for i in range(n) if not intensities[i].fixed]
        self.dependent_intensity_idx = non_fixed_I[-1] if non_fixed_I else None

        self.free_names = []
        self.bounds_lower = []
        self.bounds_upper = []

        for i in range(n):
            if not lifetimes[i].fixed:
                self.free_names.append(f"tau_{i}")
                self.bounds_lower.append(lifetimes[i].lower)
                self.bounds_upper.append(lifetimes[i].upper)

        for i in range(n):
            if not intensities[i].fixed and i != self.dependent_intensity_idx:
                self.free_names.append(f"I_{i}")
                self.bounds_lower.append(intensities[i].lower)
                self.bounds_upper.append(intensities[i].upper)

        if not t0.fixed:
            self.free_names.append("t0")
            self.bounds_lower.append(t0.lower)
            self.bounds_upper.append(t0.upper)

        if not background.fixed:
            self.free_names.append("bg")
            self.bounds_lower.append(background.lower)
            self.bounds_upper.append(background.upper)

    @property
    def n_free(self) -> int:
        return len(self.free_names)

    def initial_vector(self) -> np.ndarray:
        """Flat vector of initial values for the free parameters."""
        return self.pack(
            [p.value for p in self._lifetimes],
            [p.value for p in self._intensities],
            self._t0.value,
            self._background.value,
        )

    def pack(self, lt_vals, I_vals, t0_val, bg_val) -> np.ndarray:
        """Collect free-parameter values into the flat optimizer vector."""
        n = len(self._lifetimes)
        x = []
        for i in range(n):
            if not self._lifetimes[i].fixed:
                x.append(lt_vals[i])
        for i in range(n):
            if (not self._intensities[i].fixed
                    and i != self.dependent_intensity_idx):
                x.append(I_vals[i])
        if not self._t0.fixed:
            x.append(t0_val)
        if not self._background.fixed:
            x.append(bg_val)
        return np.array(x)

    def unpack(self, x) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Expand the flat optimizer vector to (lifetimes, intensities, t0, bg).

        Fixed parameters take their stored values; the dependent intensity
        is computed as 1 - sum(all other intensities).
        """
        n = len(self._lifetimes)
        lt_vals = np.array([p.value for p in self._lifetimes], dtype=float)
        I_vals = np.array([p.value for p in self._intensities], dtype=float)
        t0_val = self._t0.value
        bg_val = self._background.value

        idx = 0
        for i in range(n):
            if not self._lifetimes[i].fixed:
                lt_vals[i] = x[idx]
                idx += 1
        for i in range(n):
            if (not self._intensities[i].fixed
                    and i != self.dependent_intensity_idx):
                I_vals[i] = x[idx]
                idx += 1

        if self.dependent_intensity_idx is not None:
            dep = self.dependent_intensity_idx
            I_vals[dep] = 1.0 - I_vals.sum() + I_vals[dep]

        if not self._t0.fixed:
            t0_val = x[idx]
            idx += 1
        if not self._background.fixed:
            bg_val = x[idx]
            idx += 1

        return lt_vals, I_vals, t0_val, bg_val
