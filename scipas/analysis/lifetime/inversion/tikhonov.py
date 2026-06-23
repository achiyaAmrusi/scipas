import numpy as np
from scipy.optimize import nnls, minimize
from scipas.core.lifetime import PASLifetime
from scipas.analysis.lifetime.inversion import LifetimeInvert
from scipas.analysis.lifetime.inversion.utils import _response_matrix, _svd_truncate


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
    Note:
        This method is very sensitive to small shifts in the spectrum.
         If the fit is not well and the tail deviats from the measurment,
          you can move the spectrum half a time bin and it might fit correctly.
          The reason is yet to be found.
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
        D2 = (np.eye(n_tau, k=0) - 2 * np.eye(n_tau, k=1) + np.eye(n_tau, k=2))[:n_tau - 2]

        A_aug = np.vstack([response, np.sqrt(alpha) * D2])
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
        # solve for q unsing nnls
        alpha = np.exp(log_alpha[0])
        q = self._tikhonov_solution(data, response, alpha, maxiter)

        predicted = response @ q

        residuals = data - predicted
        chi_sq = residuals ** 2 / data_err **2

        return np.abs(np.sum(chi_sq) / len(data) - 1)

    def invert(self,
               pals: PASLifetime,
               bg_est: float = 0.0,
               t0_shift: float = 0.0,
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
        bg_est : background level estimated from the flat tail.
            Default 0.0.
        t0_shift : time-axis shift applied when building the response matrix.
            Shifts the model's time-zero relative to the data. Default 0.0.
        maxiter : max NNLS iterations. Defaults to 10 * n_tau.
        initial_alpha : starting alpha for analysis. Default 1e-5.
        method : scipy minimize method. Default "Powell".
        regulator_bounds : (min, max) bounds for alpha search.
        minimization_ftol : analysis convergence tolerance.

        Returns
        -------
        q : lifetime distribution over characteristic_time_grid, in
            counts·ns⁻¹ (not normalized — divide by q.sum()*dtau if needed).
        res : scipy OptimizeResult. Optimal alpha = np.exp(res.x[0]).
        """
        if maxiter is None:
            maxiter = 10 * self.characteristic_time_grid.shape[0]

        counts = pals.lifetime.counts
        net_counts = counts - bg_est
        norm = np.trapezoid(net_counts, pals.lifetime.energy)
        data = net_counts / norm
        data_err = np.sqrt(np.maximum(counts, 1)) / norm

        time_values = pals.lifetime.energy.values - t0_shift
        response = _response_matrix(
            self.characteristic_time_grid,
            time_values,
            pals.resolution
        )

        # correct the response to trapz integration
        dtau = self.characteristic_time_grid[1] - self.characteristic_time_grid[0]

        w = np.ones_like(self.characteristic_time_grid)
        w[0] *= 0.5
        w[-1] *= 0.5
        weighted_response = response * w[None,:] * dtau

        res = minimize(
            self._chi_sq,
            x0=[np.log(initial_alpha)],
            args=(data, data_err, weighted_response, maxiter),
            bounds=[(np.log(regulator_bounds[0]), np.log(regulator_bounds[1]))],
            method=method,
            options={"ftol": minimization_ftol}
        )

        alpha_opt = np.exp(res.x[0])
        q = self._tikhonov_solution(data, weighted_response, alpha_opt, maxiter)

        return q, res