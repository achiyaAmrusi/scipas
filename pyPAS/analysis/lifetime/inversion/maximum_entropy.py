import numpy as np
from scipy.optimize import minimize
from pyPAS.core.lifetime import PASLifetime
from pyPAS.analysis.lifetime.inversion import LifetimeInvert
from pyPAS.analysis.lifetime.inversion.utils import _response_matrix, _svd_truncate


class MaximalEntropyInversion(LifetimeInvert):
    """
    Inverts a positron lifetime spectrum into a lifetime distribution f(τ)
    using the Maximum Entropy Method (MELT), following Bryan (1990).

    Maximizes:  Q(f, α) = α·S(f) - L(f)

        S(f) = Σ(fᵢ - mᵢ - fᵢ log(fᵢ/mᵢ))   entropy relative to prior m (≤ 0)
        L(f) = ½·χ²                             negative log-likelihood

    f is parameterized as fᵢ = mᵢ·exp(Uᵢₜ·u) (Bryan eq. 9), restricting the
    solution to the data-visible subspace and keeping f > 0 by construction.

    Background is subtracted before normalization. Alpha and f are optimized
    jointly in log-alpha space via Powell.

    Reference: Bryan, R.K. (1990). Maximum Entropy and Bayesian Methods, 221-232.

    Note: the result distribution is normalized using np.trapezoid and not rectangular sum.
    This can cause the normalization response@f(tau) to be off 1 due to boundry singularities.
    """

    def _entropy(self, f: np.ndarray, prior: np.ndarray) -> float:
        """
        S(f|m) = Σ(fᵢ - mᵢ - fᵢ log(fᵢ/mᵢ)) ≤ 0.
        Zero only when f == m. Penalizes deviation from the prior.
        """
        f = np.clip(f, 1e-30, None)
        return np.sum(f - prior - f * np.log(f / prior))

    def _likelihood(self,
                    f: np.ndarray,
                    U: np.ndarray,
                    V: np.ndarray,
                    sigma: np.ndarray,
                    Ut: np.ndarray,
                    data: np.ndarray,
                    data_err: np.ndarray) -> float:
        """
        L(f) = ½·χ² = ½ Σ (F_k - D_k)² / σ_k²

        Forward model via truncated SVD: F = V @ diag(σ) @ Uᵀ @ f.
        data and data_err are precomputed in invert — not recomputed here.

        Parameters
        ----------
        f        : current lifetime distribution, shape (N_tau,).
        U, V, sigma, Ut : truncated SVD components (Bryan convention).
        data     : background-subtracted normalized spectrum, shape (N_time,).
        data_err : per-channel Poisson uncertainty on data, shape (N_time,).
        """
        F = V @ (sigma * (Ut @ f))
        return 0.5 * np.sum((data - F) ** 2 / data_err ** 2)

    def _Q(self, f, alpha, prior, U, V, sigma, Ut, data, data_err) -> float:
        """Q = α·S(f) - L(f), to be maximized over f and alpha."""
        return (alpha * self._entropy(f, prior)
                - self._likelihood(f, U, V, sigma, Ut, data, data_err))

    def _f_from_u(self, u: np.ndarray, prior: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Bryan eq. (9): fᵢ = mᵢ · exp(Uᵢₜ · u).
        Keeps f > 0 by construction and restricts solution to the
        data-visible subspace, reducing optimization from N_tau to s variables.
        """
        return prior * np.exp(U @ u)

    def _optimize_f(self, alpha, prior, U, V, sigma, Ut, data, data_err, u0):
        """
        Maximize Q over u at fixed alpha via L-BFGS-B.
        Operates in the s-dimensional subspace (s << N_tau).

        Parameters
        ----------
        alpha          : fixed regularization parameter.
        prior          : prior model, shape (N_tau,).
        U, V, sigma, Ut: truncated SVD components.
        data, data_err : precomputed normalized spectrum and uncertainties.
        u0             : warm-start vector, shape (s,).

        Returns
        -------
        f_hat : optimal distribution at this alpha, shape (N_tau,).
        u_opt : optimal subspace vector, shape (s,).
        """

        def neg_Q(u):
            return -self._Q(self._f_from_u(u, prior, U),
                            alpha, prior, U, V, sigma, Ut, data, data_err)

        result = minimize(neg_Q, u0, method='L-BFGS-B')
        return self._f_from_u(result.x, prior, U), result.x

    def _optimize(self, prior, U, V, sigma, Ut, data, data_err,
                  initial_alpha, alpha_bounds, ftol, maxiter):
        """
        Jointly optimize f and alpha via Powell in log-alpha space.
        At each Powell step, f is optimized at the proposed alpha via L-BFGS-B.
        u is warm-started between Powell steps for efficiency.

        Parameters
        ----------
        prior          : prior model, shape (N_tau,).
        U, V, sigma, Ut: truncated SVD components (Bryan convention).
        data, data_err : precomputed normalized spectrum and uncertainties.
        initial_alpha  : starting point for alpha search.
        alpha_bounds   : (min, max) search range for alpha.
        ftol           : Powell convergence tolerance.
        maxiter        : max Powell iterations.

        Returns
        -------
        alpha_opt : optimal regularization parameter.
        f_hat     : optimal lifetime distribution, shape (N_tau,).
        """
        u = np.zeros(len(sigma))

        def objective(log_alpha):
            nonlocal u
            alpha = np.exp(log_alpha)
            f_hat, u = self._optimize_f(alpha, prior, U, V, sigma, Ut, data, data_err, u)
            return -self._Q(f_hat, alpha, prior, U, V, sigma, Ut, data, data_err)

        result = minimize(
            objective,
            x0=np.log(initial_alpha),
            method='Powell',
            bounds=[(np.log(alpha_bounds[0]), np.log(alpha_bounds[1]))],
            options={"ftol": ftol, "maxiter": maxiter}
        )

        alpha_opt = np.exp(result.x.item())
        f_hat, _ = self._optimize_f(alpha_opt, prior, U, V, sigma, Ut, data, data_err, u)
        return alpha_opt, f_hat

    def invert(self,
               pals: PASLifetime,
               bg_est: float = 0.0,
               t0_shift: float = 0.0,
               noise_level: float = 1e-3,
               initial_alpha: float = 1e-3,
               alpha_bounds: tuple = (1e-10, 1e2),
               prior_model: np.ndarray = None,
               minimization_ftol: float = 1e-6,
               maxiter: int = None) -> tuple[float, np.ndarray]:
        """
        Invert a positron lifetime spectrum into a distribution f(τ).

        Parameters
        ----------
        pals              : measured lifetime spectrum with resolution function.
        bg_est            : per-channel background (counts/channel) from tail
                            estimate: np.mean(counts[time > tail_start]).
                            Subtracted before normalization. Default 0.
        t0_shift          : time-axis shift applied when building the response
                            matrix. Shifts the model's time-zero relative to the
                            data. Default 0.0.
        noise_level       : SVD truncation threshold relative to the largest
                            singular value. Controls rank s. Default 1e-3.
        initial_alpha     : starting alpha for the analysis. Default 1e-3.
        alpha_bounds      : (min, max) search range for alpha. Default (1e-10, 1e2).
        prior_model       : prior over tau grid. Default flat (uniform).
        minimization_ftol : Powell convergence tolerance. Default 1e-6.
        maxiter           : max Powell iterations. Default 10 * n_tau.

        Returns
        -------
        alpha_opt : optimal regularization parameter.
        f_hat     : lifetime distribution over characteristic_time_grid.
        """
        n_tau = len(self.characteristic_time_grid)
        dtau = self.characteristic_time_grid[1] - self.characteristic_time_grid[0]

        if maxiter is None:
            maxiter = 10 * n_tau
        if prior_model is None:
            prior_model = np.ones(n_tau) / np.trapezoid(np.ones(n_tau), self.characteristic_time_grid)

        counts = pals.lifetime.counts
        net_counts = counts - bg_est
        norm = np.trapezoid(net_counts, pals.lifetime.energy)
        data = net_counts / norm
        data_err = np.sqrt(np.maximum(counts, 1)) / norm

        time_values = pals.lifetime.energy.values - t0_shift
        response = _response_matrix(self.characteristic_time_grid, time_values, pals.resolution)
        w = np.ones(n_tau) * dtau
        w[0] *= 0.5
        w[-1] *= 0.5
        V, sigma, Ut = _svd_truncate(response * w[None, :], noise_level)
        U = Ut.T  # (N_tau × s) — Bryan's U, tau space

        alpha_opt, f_hat = self._optimize(
            prior=prior_model,
            U=U, V=V, sigma=sigma, Ut=Ut,
            data=data, data_err=data_err,
            initial_alpha=initial_alpha,
            alpha_bounds=alpha_bounds,
            ftol=minimization_ftol,
            maxiter=maxiter
        )

        return alpha_opt, f_hat