import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from pyPAS.core.lifetime import PASLifetime
from pyPAS.analysis.lifetime.inversion import LifetimeInvert
from pyPAS.analysis.lifetime.inversion.utils import _response_matrix


class GPRegression(LifetimeInvert):
    """
    Inverts a positron lifetime spectrum into a lifetime distribution f(τ)
    using Gaussian Process regression in log-space.

    Models g(τ) = log(c(τ)) as a GP with RBF kernel on log(τ), so the
    distribution is positive by construction. A free amplitude N separates
    total counts from the distribution shape:

        counts(t) = N * RM @ c,    c = exp(g),  sum(c) ≈ 1

    where RM columns are sum-normalized response functions (IRF-convolved
    exponential decays, one per τ on the grid).

    The posterior over g is approximated by Laplace: the MAP estimate plus
    a Gaussian with covariance (W + K⁻¹)⁻¹, where W is the Gauss-Newton
    Hessian of the data term.

    Kernel hyperparameters (length scale ℓ in log(τ), log-amplitude) are
    selected by maximizing the Laplace approximation of the marginal
    likelihood (evidence),

        -log Z ≈ ½χ²(g*) + ½(g*-m)ᵀK⁻¹(g*-m) + ½ logdet(I + K·W),

    evaluated on a grid that is traversed smooth-to-flexible with warm
    starts (homotopy), which keeps the MAP optimizations well-conditioned.
    The amplitude N is profiled at its MAP value (its one-dimensional
    Laplace factor varies slowly with the hyperparameters and is dropped).

    Unlike Tikhonov and MELT, the GP provides calibrated pointwise
    uncertainty on f(τ) via the delta method: σ_f ≈ f · σ_g.
    """

    @staticmethod
    def _rbf_kernel(log_tau, log_amplitude, length_scale):
        """RBF kernel: K = exp(log_amp) * exp(-d²/(2ℓ²))."""
        sq_dist = (log_tau[:, None] - log_tau[None, :]) ** 2
        return np.exp(log_amplitude) * np.exp(-0.5 * sq_dist / length_scale ** 2)

    def _build_rm(self, pals, t0_shift):
        """Build column-sum-normalized response matrix (n_time, n_tau)."""
        tau_grid = self.characteristic_time_grid
        time_values = pals.lifetime.energy.values - t0_shift
        response = _response_matrix(tau_grid, time_values, pals.resolution)

        col_sums = response.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        return response / col_sums[None, :]

    def _find_map(self, theta0, K_cf, RM, counts, counts_var, prior_mean):
        """MAP for the positive GP. theta = [log_N, g_0, ..., g_{n-1}].

        Minimizes ½χ² + ½(g-m)ᵀK⁻¹(g-m) with analytic gradients.
        """
        inv_var = 1.0 / counts_var

        def objective_and_grad(theta):
            log_N = theta[0]
            g = theta[1:]
            N = np.exp(np.clip(log_N, -50.0, 50.0))
            c = np.exp(np.clip(g, -50.0, 50.0))

            pred = RM @ c
            residual = counts - N * pred
            wres = residual * inv_var

            diff = g - prior_mean
            K_inv_diff = cho_solve(K_cf, diff)

            value = 0.5 * (residual @ wres + diff @ K_inv_diff)

            grad = np.empty_like(theta)
            grad[0] = -N * (pred @ wres)
            grad[1:] = -N * c * (RM.T @ wres) + K_inv_diff
            return value, grad

        return minimize(
            objective_and_grad, theta0, jac=True,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-9}
        )

    @staticmethod
    def _gauss_newton_w(g_map, N_map, RM, counts_var):
        """Gauss-Newton Hessian of the data term w.r.t. g at the MAP."""
        c = np.exp(g_map)
        A = RM * c[None, :]
        return N_map ** 2 * A.T @ (A / counts_var[:, None])

    def _neg_log_evidence(self, theta_map, K, K_cf, RM, counts, counts_var,
                          prior_mean):
        """Laplace approximation of -log p(data | ℓ, amplitude).

        -log Z ≈ ½χ² + ½ quad + ½ logdet(I + K·W).
        The logdet term is the Occam penalty: it grows with kernel
        amplitude and with the effective degrees of freedom (small ℓ),
        so the evidence balances data fit against model flexibility.
        """
        log_N = theta_map[0]
        g = theta_map[1:]
        N = np.exp(log_N)
        c = np.exp(g)

        model = N * (RM @ c)
        chi2 = np.sum((counts - model) ** 2 / counts_var)
        diff = g - prior_mean
        quad = diff @ cho_solve(K_cf, diff)

        W = self._gauss_newton_w(g, N, RM, counts_var)
        B = np.eye(len(g)) + K @ W
        sign, logdet_B = np.linalg.slogdet(B)
        if sign <= 0:
            return np.inf

        return 0.5 * (chi2 + quad + logdet_B)

    def invert(self, pals: PASLifetime, bg_est: float = 0.0,
               t0_shift: float = 0.0,
               log_amplitude: float = 3.0, length_scale: float = 0.3,
               optimize_hyperparams: bool = True) -> tuple[np.ndarray, dict]:
        """
        Invert a lifetime spectrum into a distribution f(τ) via GP regression.

        Parameters
        ----------
        pals : PASLifetime
            Measured lifetime spectrum with resolution function.
        bg_est : float
            Per-channel background estimate. Default 0.0.
        t0_shift : float
            Time-axis shift for the response matrix. Default 0.0.
        log_amplitude : float
            Log kernel amplitude: K = exp(log_amp) * RBF. Prior variance
            of g = log(c) around the uniform prior mean. Default 3.0.
        length_scale : float
            RBF length scale in log(τ) space — the smoothness of the
            recovered distribution. Default 0.3.
        optimize_hyperparams : bool
            If True, select (length_scale, log_amplitude) by maximizing
            the Laplace evidence over a warm-started grid. Default True.

        Returns
        -------
        f : np.ndarray
            Posterior mean of f(τ) (density) over characteristic_time_grid.
        metadata : dict
            'posterior_std', 'posterior_mean_log', 'posterior_cov_log',
            'N', 'log_amplitude', 'length_scale', 'neg_log_evidence',
            'optimize_result'.
        """
        tau_grid = self.characteristic_time_grid
        n_tau = len(tau_grid)
        log_tau = np.log(tau_grid)
        dtau = tau_grid[1] - tau_grid[0]

        raw_counts = pals.lifetime.counts
        net_counts = raw_counts - bg_est
        counts_var = np.maximum(raw_counts, 1.0)

        RM = self._build_rm(pals, t0_shift)

        prior_mean = np.full(n_tau, np.log(1.0 / n_tau))
        log_N0 = np.log(max(net_counts.sum(), 1.0))

        def make_k(amp, ell):
            K = self._rbf_kernel(log_tau, amp, ell)
            # relative jitter keeps Cholesky stable across amplitudes
            K[np.diag_indices_from(K)] += 1e-6 * np.exp(amp)
            return K

        theta = np.concatenate([[log_N0], prior_mean])

        evidence_grid = []
        if optimize_hyperparams:
            # smooth → flexible: each MAP warm-starts from the previous one
            ell_grid = [1.2, 0.8, 0.55, 0.4, 0.28, 0.2, 0.14, 0.1, 0.07, 0.05]
            amp_grid = [1.0, 2.0, 3.0, 4.0, 5.0]

            candidates = []
            for ell in ell_grid:
                for amp in amp_grid:
                    K = make_k(amp, ell)
                    K_cf = cho_factor(K)
                    res = self._find_map(
                        theta, K_cf, RM, net_counts, counts_var, prior_mean
                    )
                    theta = res.x
                    nle = self._neg_log_evidence(
                        res.x, K, K_cf, RM, net_counts, counts_var, prior_mean
                    )
                    evidence_grid.append((ell, amp, nle))
                    candidates.append((nle, amp, ell, res.x.copy()))

            # The evidence often plateaus below the resolution limit of the
            # data; differences within ~1 nat are not significant. Among
            # statistically indistinguishable models, prefer the smoothest
            # (largest ℓ) — an Occam tie-break.
            nle_min = min(c[0] for c in candidates)
            plateau = [c for c in candidates if c[0] <= nle_min + 1.0]
            ell_best = max(c[2] for c in plateau)
            best = min(
                (c for c in plateau if c[2] == ell_best), key=lambda c: c[0]
            )

            neg_log_evidence, log_amplitude, length_scale, theta_best = best

            # polish at the selected hyperparameters
            K = make_k(log_amplitude, length_scale)
            K_cf = cho_factor(K)
            opt_result = self._find_map(
                theta_best, K_cf, RM, net_counts, counts_var, prior_mean
            )
        else:
            K = make_k(log_amplitude, length_scale)
            K_cf = cho_factor(K)
            opt_result = self._find_map(
                theta, K_cf, RM, net_counts, counts_var, prior_mean
            )
            neg_log_evidence = self._neg_log_evidence(
                opt_result.x, K, K_cf, RM, net_counts, counts_var, prior_mean
            )

        N_map = np.exp(opt_result.x[0])
        g_map = opt_result.x[1:]
        c_map = np.exp(g_map)
        f_density = c_map / dtau

        # Laplace posterior covariance of g: (W + K⁻¹)⁻¹
        W = self._gauss_newton_w(g_map, N_map, RM, counts_var)
        K_inv = cho_solve(K_cf, np.eye(n_tau))
        H = W + K_inv
        try:
            cov_g = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov_g = np.linalg.pinv(H)
        var_g = np.diag(cov_g)
        posterior_std_density = f_density * np.sqrt(np.maximum(var_g, 0))

        metadata = {
            'posterior_std': posterior_std_density,
            'posterior_mean_log': g_map,
            'posterior_cov_log': cov_g,
            'N': N_map,
            'log_amplitude': log_amplitude,
            'length_scale': length_scale,
            'neg_log_evidence': neg_log_evidence,
            'evidence_grid': evidence_grid,
            'optimize_result': opt_result,
        }

        return f_density, metadata
