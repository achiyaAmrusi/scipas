import numpy as np
from scipy.optimize import nnls, minimize
from pyPAS.core.lt import PASLifetime
from pyPAS.optimizer.lifetime.inversion import LifetimeInvert
from pyPAS.optimizer.lifetime.inversion.utils import _response_matrix, _svd_truncate


class TikhonovRegularization(LifetimeInvert):
    """
    Inverts a positron lifetime spectrum into a lifetime distribution q(τ)
    using Tikhonov regularization with automatic alpha selection.

    Solves:
        min ||R·dτ·q - (s - bg)||² + α·||D²·q||²,  q >= 0

    where R is the response matrix, s is the raw count spectrum, bg is the
    per-channel background estimate, and D² is the second-order finite
    difference operator penalizing curvature.

    Alpha is selected by minimizing |χ²/N - 1| (discrepancy principle).
    """

    def _tikhonov_solution(self,
                           net_counts: np.ndarray,
                           response: np.ndarray,
                           alpha: float,
                           maxiter: int = None) -> np.ndarray:
        """
        Solve the regularized NNLS for fixed alpha.

        Augmented system:
            [R·dτ   ] q = [s - bg]
            [√α·D²  ]     [0     ]

        Parameters
        ----------
        net_counts : background-subtracted spectrum (may contain negatives).
        response : response matrix, shape (n_time, n_tau).
        alpha : regularization strength.
        maxiter : max NNLS iterations.

        Returns
        -------
        q : lifetime distribution over characteristic_time_grid.
        """
        n_tau = len(self.characteristic_time_grid)
        dtau = self.characteristic_time_grid[1] - self.characteristic_time_grid[0]
        D2 = (np.eye(n_tau, k=0) - 2 * np.eye(n_tau, k=1) + np.eye(n_tau, k=2))[:n_tau - 2]

        A_aug = np.vstack([response * dtau, np.sqrt(alpha) * D2])
        b_aug = np.concatenate([net_counts, np.zeros(n_tau - 2)])

        q, _ = nnls(A_aug, b_aug, maxiter=maxiter)
        return q

    def _chi_sq(self,
                log_alpha: np.ndarray,
                data: np.ndarray,
                data_err: np.ndarray,
                response: np.ndarray,
                maxiter: int,
                ) -> float:
        """
        Discrepancy-principle target: |χ²/N - 1|.

        Finds alpha where the forward model residual matches the noise level.
        Works in raw count space so Poisson errors are natural.

        Parameters
        ----------
        log_alpha : length-1 array, log of regularization parameter.
        counts : raw measured counts (for Poisson error estimate).
        net_counts : background-subtracted counts passed to NNLS.
        response : response matrix.
        bg_est : background level estimated from the flat tail
        maxiter : max NNLS iterations.
        error : if True, use Poisson-weighted chi-squared.
        """
        dtau = self.characteristic_time_grid[1] - self.characteristic_time_grid[0]
        # solve for q unsing nnls
        alpha = np.clip(np.exp(log_alpha[0]), 1e-12, 1e-1)
        q = self._tikhonov_solution(data, response, alpha, maxiter)

        predicted = response @ q * dtau

        residuals = data - predicted
        chi_sq = residuals ** 2 / data_err **2

        return np.abs(np.sum(chi_sq) / len(data) - 1)

    def invert(self,
               pals: PASLifetime,
               bg_est: float = 0.0,
               noise_level: float = 1e-3,
               maxiter: int = None,
               initial_alpha: float = 1e-5,
               method: str = "Powell",
               regulator_bounds: tuple = (1e-10, 1e-1),
               minimization_ftol: float = 1e-6) -> tuple[np.ndarray, object]:
        """
        Invert a lifetime spectrum into a distribution q(τ).

        Parameters
        ----------
        pals : measured lifetime spectrum with resolution function.
        bg_est : background level estimated from the flat tail
        Default 0.0
        maxiter : max NNLS iterations. Defaults to 10 * n_tau.
        initial_alpha : starting alpha for optimizer. Default 1e-5.
        method : scipy minimize method. Default "Powell".
        regulator_bounds : (min, max) bounds for alpha search.
        minimization_ftol : optimizer convergence tolerance.
        error : if True, use Poisson-weighted chi-squared.
        svd_truncate : if given, truncate SVD of response at this threshold.

        Returns
        -------
        q : lifetime distribution over characteristic_time_grid, in
            counts·ns⁻¹ (not normalized — divide by q.sum()*dtau if needed).
        res : scipy OptimizeResult. Optimal alpha = np.exp(res.x[0]).
        """
        if maxiter is None:
            maxiter = 10 * self.characteristic_time_grid.shape[0]

        # Background subtraction then normalization — bg excluded from norm
        counts = pals.lifetime.counts
        net_counts = counts - bg_est
        norm = np.trapz(net_counts, pals.lifetime.energy)
        data = net_counts / norm
        data_err = np.sqrt(np.maximum(counts, 1)) / norm  # Poisson error on raw counts

        response = _response_matrix(
            self.characteristic_time_grid,
            pals.lifetime.energy.values,
            pals.resolution
        )

#        Up, sp, Vtp = _svd_truncate(response, noise_level)
#        truncate_response = Up @ np.diag(sp) @ Vtp

        res = minimize(
            self._chi_sq,
            x0=[np.log(initial_alpha)],
            args=(data, data_err, response, maxiter),
            bounds=[(np.log(regulator_bounds[0]), np.log(regulator_bounds[1]))],
            method=method,
            options={"ftol": minimization_ftol}
        )

        alpha_opt = np.exp(res.x[0])
        q = self._tikhonov_solution(data, response, alpha_opt, maxiter)

        return q, res