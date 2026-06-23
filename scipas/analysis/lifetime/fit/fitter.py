import numpy as np
from scipy.optimize import least_squares

from scipas.core.lifetime import PASLifetime
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.fit.parameters import FitParameter, ParameterMap
from scipas.analysis.lifetime.fit.result import FitResult


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

    def _forward_model(self, time, lt_vals, I_vals, t0_val, bg_val,
                        resolution, total_signal):
        """
        Compute the predicted counts per bin.

        Model: N(t_i) = total_signal * dt * shape(t_i) + bg_val

        where shape(t) = IRF ⊗ Σ (I_i/τ_i) exp(-(t-t0)/τ_i), normalized
        so that ∫ shape dt = 1.

        The time-zero shift t0 is applied to the *convolved* shape by
        interpolation rather than by truncating the decay support to the
        grid. Convolving the decay with an IRF shifted by t0 is exactly
        equivalent to shifting the whole convolved shape by t0, so this is
        physically identical — but it keeps the model smooth (and therefore
        differentiable) in t0. Truncating the support at `time >= t0_val`
        instead quantizes the rising edge to the grid spacing, which zeroes
        out the finite-difference Jacobian column for t0 and stalls the fit.
        """
        decay = np.zeros_like(time, dtype=float)
        mask = time >= 0.0
        t_nonneg = time[mask]

        for tau, intensity in zip(lt_vals, I_vals):
            decay[mask] += (intensity / tau) * np.exp(-t_nonneg / tau)

        convolved = resolution.convolve(decay, time)
        norm = np.trapezoid(convolved, time)
        if norm > 0:
            convolved = convolved / norm

        if t0_val != 0.0:
            convolved = np.interp(time - t0_val, time, convolved,
                                  left=0.0, right=0.0)

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

        for lp in lifetimes:
            if lp.lower == -np.inf:
                lp.lower = 1e-10
            if lp.upper == np.inf:
                lp.upper = 1e6
        for ip in intensities:
            if ip.lower == -np.inf:
                ip.lower = 0.0
            if ip.upper == np.inf:
                ip.upper = 1.0

        time = pals.lifetime.energy.values
        counts = pals.lifetime.counts
        sigma = np.sqrt(np.maximum(counts, 1.0))

        pmap = ParameterMap(lifetimes, intensities, t0, background)
        if pmap.n_free == 0:
            raise ValueError("No free parameters — nothing to fit")

        def residual_fn(x):
            lt_vals, I_vals, t0_val, bg_val = pmap.unpack(x)
            total_signal = counts.sum() - bg_val * len(counts)
            predicted = self._forward_model(
                time, lt_vals, I_vals, t0_val, bg_val,
                pals.resolution, total_signal
            )
            return (counts - predicted) / sigma

        result = least_squares(
            residual_fn, pmap.initial_vector(),
            bounds=(pmap.bounds_lower, pmap.bounds_upper),
            method=method,
            max_nfev=10000,
        )

        lt_fit, I_fit, t0_fit, bg_fit = pmap.unpack(result.x)
        total_signal_fit = counts.sum() - bg_fit * len(counts)
        fitted_total = self._forward_model(
            time, lt_fit, I_fit, t0_fit, bg_fit,
            pals.resolution, total_signal_fit
        )

        chi2 = np.sum(((counts - fitted_total) / sigma) ** 2)
        dof = len(counts) - pmap.n_free
        reduced_chi2 = chi2 / dof if dof > 0 else np.inf

        # Covariance from Jacobian
        try:
            J = result.jac
            JtJ = J.T @ J
            cov = np.linalg.inv(JtJ) * reduced_chi2
        except np.linalg.LinAlgError:
            cov = np.full((pmap.n_free, pmap.n_free), np.nan)

        param_errors = {}
        for i, name in enumerate(pmap.free_names):
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
            n_free=pmap.n_free,
            covariance=cov,
            parameter_errors=param_errors,
            fitted_spectrum=fitted_total,
            residuals=weighted_residuals,
            success=result.success,
            message=result.message,
        )
