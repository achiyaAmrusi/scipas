import numpy as np
import xarray as xr
from scipy.optimize import nnls, minimize
from pyPAS.core.lt import PASLifetime
from pyPAS.optimizer.lifetime.inversion import LifetimeInvert
from pyPAS.optimizer.lifetime.inversion.utils import _response_matrix


class TikhonovRegularization(LifetimeInvert):
    """
    Inverts a positron lifetime spectrum into a lifetime distribution q(τ)
    using Tikhonov regularization with automatic alpha selection.

    The inversion solves:
        min ||R@q - s||^2 + α||D^2@q||^2,  q >= 0

    where R is the response matrix, s is the normalized spectrum,
    D^2 is the second-order finite difference operator (penalizes curvature),
    and α controls the smoothness tradeoff.

    Alpha is selected automatically by minimizing the chi-squared residual
    between the forward model and the data, searched in log space.
    """

    def _tikhonov_solution(self,
                           normlized_pals: np.ndarray,
                           response: np.ndarray,
                           alpha: float,
                           maxiter=None) -> np.ndarray:
        """
        Solve the regularized inversion for a fixed alpha via augmented NNLS.

        The response matrix approximates the continuous integral ∫ R(t,τ) q(τ) dτ,
        making q a true probability density over τ rather than a per-bin weight.

        The augmented system is:
            [R·dτ        ] q = [s]
            [√α · D²     ]     [0]
        (first equation is the optimization and the second is the penalty)

        Parameters
        ----------
        normlized_pals : np.ndarray
            Measured spectrum normalized to unit integral.
        response : np.ndarray
            Response matrix of shape (n_time, n_tau).
        alpha : float
            Regularization strength.
        maxiter : int, optional
            Max iterations for NNLS solver.

        Returns
        -------
        q : np.ndarray
            Lifetime distribution over the characteristic time grid.
        """
        n_tau = len(self.characteristic_time_grid)
        dtau = self.characteristic_time_grid[1] - self.characteristic_time_grid[0]

        D2 = (np.eye(n_tau, k=0) - 2 * np.eye(n_tau, k=1) + np.eye(n_tau, k=2))[:n_tau - 2]

        A_aug = np.vstack([response * dtau, np.sqrt(alpha) * D2])
        b_aug = np.concatenate([normlized_pals, np.zeros(n_tau - 2)])

        q, _ = nnls(A_aug, b_aug, maxiter=maxiter)
        return q

    def chi_sq_log(self, log_alpha, pals, response, maxiter, error=True) -> float:
        """
        Chi-squared residual between forward model and data, as a function of log(α).

        Optimized in log space so alpha stays positive and spans orders of magnitude
        naturally. If error=True, residuals are weighted by Poisson uncertainties.

        Parameters
        ----------
        log_alpha : array-like of length 1
            Log of the regularization parameter.
        pals : PASLifetime
            Measured lifetime spectrum.
        response : np.ndarray
            Response matrix.
        maxiter : int
            Max NNLS iterations.
        error : bool
            If True, use Poisson-weighted chi-squared. Default True.

        Returns
        -------
        float
            Sum of (weighted) squared residuals.
        """
        alpha = float(np.exp(log_alpha[0]))
        alpha = np.clip(alpha, 1e-12, 1e-1)

        counts = pals.lifetime.counts
        time = pals.lifetime.energy.values
        norm = pals.lifetime.integrate('energy').item()
        normlized_pals = counts / norm

        if error:
            normlized_pals_err = np.sqrt(counts) / norm

        q = self._tikhonov_solution(
            normlized_pals=normlized_pals,
            response=response,
            alpha=alpha,
            maxiter=maxiter
        )

        dtau = self.characteristic_time_grid[1] - self.characteristic_time_grid[0]
        lifetime_q = response @ q * dtau

        mask = normlized_pals > 0

        if error:
            chi_sq = (normlized_pals[mask] - lifetime_q[mask]) ** 2 / normlized_pals_err[mask] ** 2
        else:
            chi_sq = (normlized_pals[mask] - lifetime_q[mask]) ** 2

        return np.sum(chi_sq)

    def invert(self, pals: PASLifetime,
               maxiter=None,
               initial_alpha=None,
               method="Powell",
               regulator_bounds=(1e-10, 1e-1),
               minimization_ftol=1e-6,
               error: bool = True) -> tuple[np.ndarray, object]:
        """
        Invert a lifetime spectrum into a lifetime distribution q(τ).

        Automatically selects the regularization parameter α by minimizing
        the chi-squared residual in log space, then returns the final q(τ)
        at the optimal α.

        Parameters
        ----------
        pals : PASLifetime
            Measured lifetime spectrum with associated resolution function.
        maxiter : int, optional
            Max NNLS iterations. Defaults to 10 * n_tau.
        initial_alpha : float, optional
            Starting alpha for the optimizer. Defaults to 1e-5.
        method : str
            Scipy minimize method. Default "Powell".
        regulator_bounds : tuple
            (min, max) bounds for alpha search. Default (1e-10, 1e-1).
        minimization_ftol : float
            Convergence tolerance for the optimizer. Default 1e-6.
        error : bool
            If True, use Poisson-weighted chi-squared. Default True.
        Returns
        -------
        q : np.ndarray
            Recovered lifetime distribution over characteristic_time_grid.
        res : OptimizeResult
            Full scipy optimization result, including optimal alpha via np.exp(res.x[0]).
        """
        if maxiter is None:
            maxiter = 10 * self.characteristic_time_grid.shape[0]
        if initial_alpha is None:
            initial_alpha = 1e-5

        response = _response_matrix(
            self.characteristic_time_grid,
            pals.lifetime.energy.values,
            pals.resolution
        )

        res = minimize(
            self.chi_sq_log,
            x0=[np.log(initial_alpha)],
            args=(pals, response, maxiter, error),
            bounds=[(np.log(regulator_bounds[0]), np.log(regulator_bounds[1]))],
            method=method,
            options={"ftol": minimization_ftol}
        )

        alpha_opt = float(np.exp(res.x[0]))

        norm = pals.lifetime.integrate('energy').item()
        normlized_pals = pals.lifetime.counts / norm

        q = self._tikhonov_solution(
            normlized_pals=normlized_pals,
            response=response,
            alpha=alpha_opt,
            maxiter=maxiter
        )

        return q, res