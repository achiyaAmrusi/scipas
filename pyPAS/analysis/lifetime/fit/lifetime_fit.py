import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from pyPAS.core.lifetime import PASLifetime, TimeResolution
from pyPAS.model.lifetime import LifetimeModel


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


@dataclass
class FitResult:
    """
    Result of a discrete multi-exponential lifetime fit.

    Attributes
    ----------
    model : LifetimeModel
        Fitted lifetime model with normalized intensities.
    t0 : float
        Fitted time-zero.
    background : float
        Fitted background level (counts per bin).
    chi_squared : float
        Sum of squared weighted residuals.
    reduced_chi_squared : float
        chi_squared / degrees of freedom.
    n_free : int
        Number of free parameters.
    covariance : np.ndarray
        Covariance matrix of free parameters (from Jacobian).
    parameter_errors : dict[str, float]
        Standard errors keyed by parameter name.
    fitted_spectrum : np.ndarray
        Model spectrum evaluated at best-fit parameters.
    residuals : np.ndarray
        Weighted residuals (data - model) / sigma.
    success : bool
        Whether the optimizer converged.
    message : str
        Optimizer status message.
    """
    model: LifetimeModel
    t0: float
    background: float
    chi_squared: float
    reduced_chi_squared: float
    n_free: int
    covariance: np.ndarray
    parameter_errors: dict
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str


class LifetimeFitter:
    """
    Discrete multi-exponential lifetime spectrum fitter.

    Fits the model:
        N(t) = IRF(t) ⊗ [Σ_i (I_i / τ_i) · exp(-(t - t0) / τ_i) · H(t - t0)] + bg

    Each parameter (τ_i, I_i, t0, background) can be individually set as
    free (with optional bounds) or fixed to a specific value.

    Intensities are constrained to sum to 1: the last non-fixed intensity
    is computed as 1 - sum(all others) and is not an independent fit parameter.

    Uses scipy.optimize.least_squares (Levenberg-Marquardt or Trust Region
    Reflective) for Jacobian-based covariance estimation.
    """

    def _build_param_map(self, lifetimes, intensities, t0, background):
        """
        Build the mapping between the flat parameter vector and named parameters.

        Returns
        -------
        free_names : list of str — names of free parameters in order.
        pack : callable — (lifetimes, intensities, t0, bg) → flat free array.
        unpack : callable — flat free array → (lifetimes, intensities, t0, bg).
        bounds_lower, bounds_upper : lists of bounds for free params.
        dependent_intensity_idx : index of the intensity computed as remainder,
            or None if all intensities are fixed.
        """
        n = len(lifetimes)
        free_names = []
        bounds_lower = []
        bounds_upper = []

        # Determine which intensity is the dependent one (last non-fixed)
        non_fixed_I = [i for i in range(n) if not intensities[i].fixed]
        dependent_idx = non_fixed_I[-1] if non_fixed_I else None

        for i in range(n):
            if not lifetimes[i].fixed:
                free_names.append(f"tau_{i}")
                bounds_lower.append(lifetimes[i].lower)
                bounds_upper.append(lifetimes[i].upper)

        for i in range(n):
            if not intensities[i].fixed and i != dependent_idx:
                free_names.append(f"I_{i}")
                bounds_lower.append(intensities[i].lower)
                bounds_upper.append(intensities[i].upper)

        if not t0.fixed:
            free_names.append("t0")
            bounds_lower.append(t0.lower)
            bounds_upper.append(t0.upper)

        if not background.fixed:
            free_names.append("bg")
            bounds_lower.append(background.lower)
            bounds_upper.append(background.upper)

        def pack(lt_vals, I_vals, t0_val, bg_val):
            x = []
            for i in range(n):
                if not lifetimes[i].fixed:
                    x.append(lt_vals[i])
            for i in range(n):
                if not intensities[i].fixed and i != dependent_idx:
                    x.append(I_vals[i])
            if not t0.fixed:
                x.append(t0_val)
            if not background.fixed:
                x.append(bg_val)
            return np.array(x)

        def unpack(x):
            lt_vals = np.array([p.value for p in lifetimes], dtype=float)
            I_vals = np.array([p.value for p in intensities], dtype=float)
            t0_val = t0.value
            bg_val = background.value

            idx = 0
            for i in range(n):
                if not lifetimes[i].fixed:
                    lt_vals[i] = x[idx]
                    idx += 1
            for i in range(n):
                if not intensities[i].fixed and i != dependent_idx:
                    I_vals[i] = x[idx]
                    idx += 1

            if dependent_idx is not None:
                I_vals[dependent_idx] = 1.0 - I_vals.sum() + I_vals[dependent_idx]

            if not t0.fixed:
                t0_val = x[idx]
                idx += 1
            if not background.fixed:
                bg_val = x[idx]
                idx += 1

            return lt_vals, I_vals, t0_val, bg_val

        return free_names, pack, unpack, bounds_lower, bounds_upper, dependent_idx

    def _forward_model(self, time, lt_vals, I_vals, t0_val, bg_val,
                        resolution, total_signal):
        """
        Compute the predicted counts per bin.

        Model: N(t_i) = total_signal * dt * shape(t_i) + bg_val

        where shape(t) = IRF ⊗ Σ (I_i/τ_i) exp(-(t-t0)/τ_i), normalized
        so that ∫ shape dt = 1.
        """
        decay = np.zeros_like(time, dtype=float)
        mask = time >= t0_val
        t_shifted = time[mask] - t0_val

        for tau, intensity in zip(lt_vals, I_vals):
            decay[mask] += (intensity / tau) * np.exp(-t_shifted / tau)

        convolved = resolution.convolve(decay, time)
        norm = np.trapezoid(convolved, time)
        if norm > 0:
            convolved = convolved / norm

        dt = time[1] - time[0]
        return total_signal * dt * convolved + bg_val

    def fit(self,
            pals: PASLifetime,
            lifetimes: list[FitParameter],
            intensities: list[FitParameter],
            t0: FitParameter = None,
            background: FitParameter = None,
            method: str = "trf",
            ) -> FitResult:
        """
        Fit a discrete multi-exponential model to a lifetime spectrum.

        Parameters
        ----------
        pals : PASLifetime
            Measured lifetime spectrum with resolution function.
        lifetimes : list of FitParameter
            Initial guesses / fixed values for each lifetime component τ_i.
            Physical default bounds: (0, ∞).
        intensities : list of FitParameter
            Initial guesses / fixed values for each intensity I_i.
            The last non-fixed intensity is computed as 1 - sum(others).
            Physical default bounds: [0, 1].
        t0 : FitParameter, optional
            Time-zero parameter. Default: FitParameter(0.0).
        background : FitParameter, optional
            Background level (counts per bin). Default: FitParameter(0.0, lower=0.0).
        method : str
            Optimization method for least_squares. Default "trf" (Trust Region
            Reflective, supports bounds). Use "lm" for Levenberg-Marquardt
            (no bounds support).

        Returns
        -------
        FitResult
        """
        if len(lifetimes) != len(intensities):
            raise ValueError("lifetimes and intensities must have the same length")
        if len(lifetimes) == 0:
            raise ValueError("At least one component is required")

        if t0 is None:
            t0 = FitParameter(0.0)
        if background is None:
            background = FitParameter(0.0, lower=0.0)

        for i, lp in enumerate(lifetimes):
            if lp.lower == -np.inf:
                lp.lower = 1e-10
            if lp.upper == np.inf:
                lp.upper = 1e6
        for i, ip in enumerate(intensities):
            if ip.lower == -np.inf:
                ip.lower = 0.0
            if ip.upper == np.inf:
                ip.upper = 1.0

        time = pals.lifetime.energy.values
        counts = pals.lifetime.counts
        dt = time[1] - time[0]
        sigma = np.sqrt(np.maximum(counts, 1.0))

        free_names, pack, unpack, bl, bu, dep_idx = self._build_param_map(
            lifetimes, intensities, t0, background
        )

        n_free = len(free_names)
        if n_free == 0:
            raise ValueError("No free parameters — nothing to fit")

        x0 = pack(
            [p.value for p in lifetimes],
            [p.value for p in intensities],
            t0.value,
            background.value,
        )

        def residual_fn(x):
            lt_vals, I_vals, t0_val, bg_val = unpack(x)
            total_signal = counts.sum() - bg_val * len(counts)
            predicted = self._forward_model(
                time, lt_vals, I_vals, t0_val, bg_val,
                pals.resolution, total_signal
            )
            return (counts - predicted) / sigma

        result = least_squares(
            residual_fn, x0,
            bounds=(bl, bu),
            method=method,
            max_nfev=10000,
        )

        lt_fit, I_fit, t0_fit, bg_fit = unpack(result.x)
        total_signal_fit = counts.sum() - bg_fit * len(counts)
        fitted_total = self._forward_model(
            time, lt_fit, I_fit, t0_fit, bg_fit,
            pals.resolution, total_signal_fit
        )

        chi2 = np.sum(((counts - fitted_total) / sigma) ** 2)
        n_data = len(counts)
        dof = n_data - n_free
        reduced_chi2 = chi2 / dof if dof > 0 else np.inf

        # Covariance from Jacobian
        try:
            J = result.jac
            JtJ = J.T @ J
            cov = np.linalg.inv(JtJ) * reduced_chi2
        except np.linalg.LinAlgError:
            cov = np.full((n_free, n_free), np.nan)

        param_errors = {}
        for i, name in enumerate(free_names):
            param_errors[name] = np.sqrt(max(cov[i, i], 0.0))

        I_fit_clipped = np.maximum(I_fit, 0.0)
        I_sum = I_fit_clipped.sum()
        if I_sum > 0:
            I_fit_clipped = I_fit_clipped / I_sum

        fitted_model = LifetimeModel(
            name="fit",
            lifetimes=lt_fit,
            intensities=I_fit_clipped,
        )

        weighted_residuals = (counts - fitted_total) / sigma

        return FitResult(
            model=fitted_model,
            t0=t0_fit,
            background=bg_fit,
            chi_squared=chi2,
            reduced_chi_squared=reduced_chi2,
            n_free=n_free,
            covariance=cov,
            parameter_errors=param_errors,
            fitted_spectrum=fitted_total,
            residuals=weighted_residuals,
            success=result.success,
            message=result.message,
        )
