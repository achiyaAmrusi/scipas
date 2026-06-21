import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs
from scipy.optimize import least_squares
from pypas.model import Sample, Material, Layer
from pypas.transport.diffusion import profile_solver
from pypas.analysis.vedb.annihilation_fractions import compute_annihilation_fractions


class DiffusionLengthOptimization:
    """
    Optimize positron diffusion lengths in a multilayer sample to best fit
    experimentally measured lineshpae-parameter from variable-energy Doppler broadening (VEDB).
    The parametrer can be S or W.

    The optimization solves the simplified positron transport equation in layered samples:

        d²c(z)/dz² − (1 / L_eff²) * c(z) = -I(z)

    where:
        - c(z)   : positron concentration as a function of depth
        - L_eff  : effective diffusion length, L_eff = sqrt(D / λ)
        - I(z)   : positron implantation source term

    For each trial set of diffusion lengths, the class:
        1. Constructs a normalized trial sample (D=1, λ = 1/L²).
        2. Solves the positron transport equation for each implantation energy.
        3. Computes annihilation fractions per layer and surface.
        4. Estimates S-parameters via linear least squares.
        5. Minimizes weighted residuals via nonlinear least squares.

    The number of layers is inferred automatically from the initial guess sample,
    making this class applicable to single-layer, two-layer, and three-layer models
    without modification.

    Parameters
    ----------
    positron_implantation_profiles : list of xr.DataArray
        Depth profiles of implanted positrons for each beam energy [positrons/nm].
        Each profile must be normalized and have a single depth coordinate.
    s_measurement : pd.Series of uncertainties.ufloat or uarray
        Measured S-parameters with uncertainties for each beam energy.
        Index must correspond to the implantation profiles.
    initial_guess : Sample
        Initial sample defining layer geometry and material properties.
        Used to extract layer widths and initial diffusion length estimates.
    num_of_mesh_cells : int, optional
        Number of mesh points for the transport equation solver. Default is 10000.
        Higher values improve accuracy at the cost of computation time.

    Attributes
    ----------
    n_layers : int
        Number of layers, inferred from initial_guess.
    s_measurement : np.ndarray
        Nominal S-parameter values extracted from the ufloat series.
    s_measurement_dev : np.ndarray
        Standard deviations of the S-parameter measurements.

    Notes
    -----
    The trial sample uses normalized units (D=1, absorption_length=1) since
    only the diffusion length L = sqrt(D/λ) affects the annihilation profile shape.
    The absolute values of D and absorption_length do not affect the optimization result.
    """

    def __init__(self,
                 positron_implantation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells: int = 10000):

        self.positron_implantation_profiles = positron_implantation_profiles
        self.initial_sample = initial_guess
        self.s_measurement = nominal_values(s_measurement)
        self.s_measurement_dev = std_devs(s_measurement)
        self.num_of_mesh_cells = num_of_mesh_cells
        self.n_layers = len(initial_guess.layers)

    def make_sample(self, diffusion_lengths: list) -> Sample:
        """
        Construct a normalized trial sample for a given set of diffusion lengths.

        Each layer is assigned D=1 and λ = 1/L² so that the effective diffusion
        length matches the trial value. Layer geometry (start, width) is preserved
        from the initial guess. The absorption length is set to 1 (normalized)
        since it does not affect the diffusion length optimization.

        Parameters
        ----------
        diffusion_lengths : list of float
            Trial diffusion lengths [nm], one per layer.
            Must have the same length as the number of layers in the initial sample.

        Returns
        -------
        Sample
            Normalized single- or multilayer sample with the given diffusion lengths.
        """
        layers = [
            Layer(
                start=self.initial_sample.layers[i].start,
                width=self.initial_sample.layers[i].width,
                material=Material(diffusion=1, mobility=0, bulk_annihilation_rate=1 / dl ** 2)
            )
            for i, dl in enumerate(diffusion_lengths)
        ]
        return Sample(layers=layers, absorption_length=1)

    def layers_transport_solver(self, sample: Sample, implantation_profiles: list) -> np.ndarray:
        """
        Solve the positron transport equation for each implantation profile
        and return the annihilation fraction matrix.

        Parameters
        ----------
        sample : Sample
            Trial sample for which the transport equation is solved.
        implantation_profiles : list of xr.DataArray
            Positron implantation profiles, one per beam energy.

        Returns
        -------
        np.ndarray
            Annihilation fraction matrix of shape (n_profiles, n_layers + 1).
            Each row corresponds to one beam energy; columns are
            [surface, layer_0, layer_1, ...].
        """
        frac_matrix = np.zeros((len(implantation_profiles), self.n_layers + 1))
        for i, p in enumerate(implantation_profiles):
            frac_matrix[i] = compute_annihilation_fractions(
                positron_profile=profile_solver(p, sample, mesh_size=self.num_of_mesh_cells),
                sample=sample).values
        return frac_matrix

    def layer_s_value(self, frac_matrix: np.ndarray) -> np.ndarray:
        """
        Estimate the S-parameter characteristic of each annihilation channel
        via linear least squares.

        Solves:  frac_matrix @ s_layers ≈ s_measurement

        Parameters
        ----------
        frac_matrix : np.ndarray
            Annihilation fraction matrix of shape (n_profiles, n_channels).

        Returns
        -------
        np.ndarray
            Estimated S-parameter per annihilation channel [surface, layer_0, ...].
            Length is n_layers + 1.
        """
        return np.linalg.lstsq(frac_matrix, self.s_measurement, rcond=None)[0]

    def residuals(self, diffusion_lengths: np.ndarray) -> np.ndarray:
        """
        Compute weighted residuals between measured and modeled S-parameters.

        For a given trial set of diffusion lengths, solves the transport equation,
        estimates S-parameters per layer, and returns the normalized difference:

            residuals = (S_calc - S_measured) / sigma_measured

        If any estimated S-parameter per layer is unphysical (negative or >= 1),
        returns a large residual vector to penalize the optimizer.

        Parameters
        ----------
        diffusion_lengths : np.ndarray
            Trial diffusion lengths [nm], one per layer.

        Returns
        -------
        np.ndarray
            Weighted residual vector of length n_profiles.
        """
        sample = self.make_sample(diffusion_lengths)
        frac_matrix = self.layers_transport_solver(sample, self.positron_implantation_profiles)
        s_vec = self.layer_s_value(frac_matrix)
        s_calc = frac_matrix @ s_vec
        if np.any(s_vec[1:] >= 1) or np.any(s_vec <= 0):
            return np.full_like(s_calc, 1e6)
        return (s_calc - self.s_measurement) / self.s_measurement_dev

    def extract_fit_results(self, ls_results) -> list:
        """
        Extract best-fit parameters and full covariance matrix from least-squares result.

        The covariance matrix is estimated from the Jacobian via the Gauss-Newton
        approximation using SVD:

        J ≈ U S Vᵀ  →  cov ≈ Vᵀ diag(1/s²) V

        Singular values below numerical precision are discarded to regularize
        the inversion.
        Parameters
        ----------
        ls_results : OptimizeResult
            Result from `scipy.optimize.least_squares`. Must contain
            `jac` (Jacobian at solution) and `x` (best-fit parameters).

        Returns
        -------
        best_fit : np.ndarray
            Best-fit diffusion lengths [nm], shape (n_layers,).
        cov : np.ndarray
            Full covariance matrix, shape (n_layers, n_layers).
            Diagonal entries are variances; off-diagonal entries capture
            parameter correlations and should not be neglected for
            correlated layers.
        """

        J = ls_results.jac
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov = np.dot(VT.T / s ** 2, VT)
        return ls_results.x, cov

    def optimize_diffusion_length(self, bounds=None) -> tuple:
        """
        Optimize diffusion lengths for all layers simultaneously.

        Performs nonlinear least-squares minimization of the weighted residuals
        between measured and modeled S-parameters. Initial guesses are derived
        from the material properties of the initial sample.

        Parameters
        ----------
        bounds : tuple of (float, float), optional
            (lower, upper) bounds applied equally to all diffusion lengths [nm].
            Must satisfy 0 <= lower <= upper. Default is (0, 1000).

        Returns
        -------
        best_fit : np.ndarray
            Best-fit diffusion lengths [nm], shape (n_layers,).
            Ordered as [layer_0, layer_1, ...].
        cov : np.ndarray
            Full covariance matrix of shape (n_layers, n_layers).
            Diagonal entries are variances; off-diagonal entries capture
            parameter correlations. Marginal uncertainties can be obtained
            via np.sqrt(np.diag(cov)).

        Raises
        ------
        ValueError
            If lower bound exceeds upper bound.
        """
        initial_guess = [
            (m.diffusion / m.effective_annihilation_rate()) ** 0.5
            for m in (layer.material for layer in self.initial_sample.layers)
        ]

        if bounds is None:
            bounds = (0, 1000)
        elif not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("lower bound must be less than upper bound")

        lb = [bounds[0]] * self.n_layers
        ub = [bounds[1]] * self.n_layers

        ls_result = least_squares(fun=self.residuals, x0=initial_guess, bounds=(lb, ub))

        return self.extract_fit_results(ls_result)
