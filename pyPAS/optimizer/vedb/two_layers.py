import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, std_devs
from scipy.optimize import  least_squares
from pyPAS.materials import Sample, Material, Layer
from pyPAS.transport.diffusion import profile_solver
from pyPAS.transport.diffusion import annihilation_fraction_per_layer


class TwoBulkDiffusionLengthOptimization:
    """
    Optimize the bulk positron diffusion length of a two-layer materials
    (plus surface contribution) to best fit experimentally measured S-parameters.

    The optimization is based on solving a simplified effective positron transport equation:

        d²c(z)/dz² − (1 / L_eff²) * c(z) = - I(z)

    where:
        - c(z)       : positron concentration as a function of depth
        - D(z)       : diffusion coefficient
        - I(z)       : positron implantation source term
        - L_eff      : effective positron diffusion length
                       (related to bulk annihilation rate λ via L_eff = sqrt(D / λ))

    The class uses:
        1. Provided implantation profiles for multiple beam energies.
        2. Experimental S-parameter data with uncertainties.
        3. A two-layer initial materials as a starting guess.

    Workflow:
        - Construct trial samples for candidate diffusion lengths.
        - Solve the positron annihilation profile for each implantation energy.
        - Compute annihilation fractions in surface and bulk.
        - Fit these to experimental S values using least-squares regression.
        - Extract the best-fit diffusion lengths and their 1σ uncertainty.

    Parameters
    ----------
    positron_implantation_profiles : list of xarray.DataArray
        List of depth profiles of implanted positrons for each beam energy.
        Each profile should have 'x' as the depth coordinate.
    s_measurement : pandas.Series of uncertainties.ufloat
        Measured S parameters for each beam energy, including uncertainties.
        Index must correspond to the energies of the implantation profiles.
    initial_guess : Sample
        Initial guess for the materials geometry and material properties.
        Must contain one layer with a diffusion coefficient and annihilation rate.
    num_of_mesh_cells : int, optional (default=10000)
        Number of discretization points for solving the transport equation.
        Larger numbers improve accuracy but increase computation time.
    """

    def __init__(self, positron_implantation_profiles: list,
                 s_measurement: pd.Series,
                 initial_guess: Sample,
                 num_of_mesh_cells=10000):
        """
    Initialize the diffusion length optimization problem for a single bulk layer.

    Parameters
    ----------
    positron_implantation_profiles : list of xarray.DataArray
        Depth distributions of implanted positrons, one per beam energy.
        Each array must use depth ('x') as a coordinate and be normalized.
    s_measurement : pandas.Series of uncertainties.ufloat
        Experimentally measured S-parameters with uncertainties.
        Index must correspond to the implantation profiles (beam energies).
    initial_sample : Sample
        Sample object defining geometry and material parameters
        (must contain one bulk layer).
    num_of_mesh_cells : int, optional (default=10000)
        Number of mesh points used for numerical solution of the transport equation.
        """
        self.positron_implantation_profiles = positron_implantation_profiles
        self.initial_sample = initial_guess
        self.energies = s_measurement.index
        self.s_measurement = nominal_values(s_measurement)
        self.s_measurement_dev = std_devs(s_measurement)
        self.num_of_mesh_cells = num_of_mesh_cells
        if len(initial_guess.layers)!=2:
            raise ValueError("Sample should have only 2 layers")


    def make_sample(self, diffusion_length_0, diffusion_length_1):
        """
        Construct a simplified two-layer materials for a given diffusion length.

        The materials consists of:
        - One layer spanning the full length of the initial materials.
        - A material with fixed diffusion (=1), zero mobility,
          and bulk annihilation rate defined as 1 / diffusion_length².
        - Absorption length is set to 1 (normalized) (because it has no effect on the diffusion length optimization).

        Parameters
        ----------
        diffusion_length_0, diffusion_length_1 : float
            Trial positron diffusion length (in nm).


        Returns
        -------
        materials : Sample
            A new Sample object containing a single layer with the above properties.
        """
        #eff_absorbtion_length = self.initial_sample.absorbtion_length * self.initial_sample.layers[0].material.diffusion
        material_0 = Material(diffusion=1, mobility=0, bulk_annihilation_rate=1/diffusion_length_0**2)
        material_1 = Material(diffusion=1, mobility=0, bulk_annihilation_rate=1 / diffusion_length_1 ** 2)
        layer_0 = Layer(start=0, width=self.initial_sample.layers[0].width, material=material_0)
        layer_1 = Layer(start=self.initial_sample.layers[1].start, width=self.initial_sample.layers[1].width, material=material_1)
        return Sample(layers=[layer_0, layer_1], absorbtion_length=1)

    def _profile_to_fractions(self, positron_profile, sample):
        """
        Extract surface and bulk annihilation fractions from a positron profile (After diffusion).
        This is a lightweight wrapper around
        `annihilation_fraction_per_layer`, which computes annihilation
        probabilities for all layers.

        Parameters
        ----------
        positron_profile : xr.DataArray
            Positron distribution across depth
        sample : Sample
            The layered materials object describing geometry and
            material annihilation rates.

        Returns
        -------
        fractions : list of float
            [surface_fraction, bulk_fraction]
            - surface_fraction : probability of annihilation at the surface
            - bulk_fraction : probability of annihilation in the first bulk layer
        """
        annihilation_fraction = annihilation_fraction_per_layer(positron_profile, sample)
        surface = annihilation_fraction.sel(layer='surface').item()
        bulk_0 = annihilation_fraction.sel(layer='layer_0').item()
        bulk_1 = annihilation_fraction.sel(layer='layer_1').item()
        return [surface, bulk_0, bulk_1]

    def layers_transport_solver(self, sample, positron_implantation_profiles):
        """
        Transport solver for annihilation fractions in a layered materials.
        For each positron implantation profile, solves the positron
        transport equation in the materials and computes annihilation
        fractions in surface and bulk regions.

        Parameters
        ----------
        sample : Sample
            The materials object containing material layers and properties.
        positron_implantation_profiles : list of xr.DataArray
            Each profile defines the depth distribution of implanted
            positrons (normalized).

        Returns
        -------
        annihilation_channel_rate_matrix : np.ndarray
            Shape (n_profiles, 2). Each row contains:
            [fraction_surface, fraction_bulk]
            corresponding to annihilation in the surface and first bulk layer.

        """
        # change to not dynamic memory
        frac_matrix = np.zeros((len(positron_implantation_profiles),len(sample.layers)+1))
        for i, p in enumerate(positron_implantation_profiles):
            frac_matrix[i] = self._profile_to_fractions(
                profile_solver(p, sample, mesh_size=self.num_of_mesh_cells), sample)
        return frac_matrix

    def s_value_per_layer(self, annihilation_channel_rate_matrix):
        """
        Estimate S-parameters for each annihilation layer via linear least square.

        Parameters
        ----------
        annihilation_channel_rate_matrix : np.ndarray
            Matrix of annihilation fractions.
            Shape: (n_profiles, n_channels)
            - Each row corresponds to an implantation profile
            - Each column corresponds to a channel (surface, bulk_0, etc.)

        Returns
        -------
        s_layer : np.ndarray
            Estimated S-parameter for each annihilation channel (length = n_channels).
        """
        return np.linalg.lstsq(annihilation_channel_rate_matrix, self.s_measurement, rcond=None)[0]


    def s_parameter_calculation(self, diffusion_length_0, diffusion_length_1):
        """
        For given materials, given the materials parameters, function calculate the expected S parameter per energy

        Parameters
        ----------
        diffusion_length_0, diffusion_length_1 : float
            Trial positron diffusion length (in nm).

        Returns
        -------
        s_sample: np.ndarray
        the expected s parameters
        """
        sample = self.make_sample(diffusion_length_0, diffusion_length_1)
        # find s_parm using linear regression
        annihilation_channel_rate_matrix = self.layers_transport_solver(sample, self.positron_implantation_profiles)
        s_vec = self.s_value_per_layer(annihilation_channel_rate_matrix)
        s_sample = annihilation_channel_rate_matrix @ s_vec

        if np.any(s_vec[1:] >= 1) or np.any(s_vec<=0) :
            # if the s value is above 1 make the result high
            return np.full_like(s_sample, 1e6)
        return s_sample

    def residuals(self, diffusion_lengths: np.array) -> np.ndarray:
        """
        Residual function for least-squares optimization of the positron diffusion lengths.

        This computes the weighted difference between measured and calculated S-parameters
        for a given trial diffusion length.

        Parameters
        ----------
        diffusion_lengths : np.ndarray
        [diffusion_length_0, diffusion_length_1]
            Trial positron diffusion length (in nm).

        Returns
        -------
        np.ndarray
            Array of normalized residuals:
            (S_calc - S_measured) / sigma_measured
            where `sigma_measured` is the experimental uncertainty.
        """
        s_calc = self.s_parameter_calculation(diffusion_lengths[0], diffusion_lengths[1])
        return (s_calc - self.s_measurement) / self.s_measurement_dev  # weighted residuals

    def extract_error(self, ls_results):
        """
        Extract parameter fit and its uncertainty from scipy.least_squares.
        The method uses the Jacobian matrix from the fit to estimate the covariance matrix (Gauss-Newton)
        via singular value decomposition (SVD). The square root of the diagonal
        element corresponding to the first parameter gives its standard error.

        Parameters
        ----------
        ls_results : OptimizeResult
            Result object returned by `scipy.optimize.least_squares`.
            Must contain attributes:
            - `jac` : Jacobian matrix at the solution.
            - `x`   : Best-fit parameter values.

        Returns
        -------
        ufloat
            The best-fit value of the first parameter with its 1-sigma uncertainty.

        Notes
        -----
        - The method does not take into account the covariance of the two diffusion length
        """
        # Approximate the covariance (Gauss-Newton)
        J = ls_results.jac
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov = np.dot(VT.T / s ** 2, VT)  # already correctly scaled if residuals are normalized

        best_fits = ls_results.x
        errs = np.sqrt(np.diag(cov))  # 1-sigma uncertainties for each parameter

        return [ufloat(val, err) for val, err in zip(best_fits, errs)]

    def optimize_diffusion_length(self, bounds=None):
        """
        Optimize the diffusion length for a two-layer materials.
        Performs nonlinear least-squares minimization of the residuals
        between measured and modeled S-parameters, with respect to the
        diffusion lengths.

        Parameters
        ----------
        bounds : tuple of (float, float), optional
            Lower and upper bounds for the diffusion length.
            Default is (0, 1 Micron). Must satisfy 0 <= lower <= upper.
            The bounds are the same for all diffusion length !

        Returns
        -------
        dict[str, ufloat]
            Mapping of layer name ("layer_0") to the optimized diffusion
            length with its 1-sigma uncertainty.
        """
        material_0 = (self.initial_sample.layers[0]).material
        material_1 = (self.initial_sample.layers[1]).material
        # The effective rate is the inverse of the diffusion length squared
        initial_diffusion_length_layer_0 = (material_0.diffusion/material_0.effective_annihilation_rate())**0.5
        initial_diffusion_length_layer_1 = (material_1.diffusion / material_1.effective_annihilation_rate()) ** 0.5
        initial_guess = [initial_diffusion_length_layer_0, initial_diffusion_length_layer_1]

        # set bound of the effective diffusion and the surface_capture_rate
        if bounds is None:
            bounds = [0, 1000]
        elif not 0<=bounds[0]<=bounds[1]:
            raise ValueError("lower bound needs to be lower than the upper bound")

        bounds = ([bounds[0], bounds[0]], [bounds[1], bounds[1]])
        # optimization
        ls_result = least_squares(fun=self.residuals, x0=initial_guess, bounds=bounds)

        # results extraction
        diffusion_lengths = self.extract_error(ls_result)

        return {'layer_0':diffusion_lengths[0], 'layer_1':diffusion_lengths[1]}
