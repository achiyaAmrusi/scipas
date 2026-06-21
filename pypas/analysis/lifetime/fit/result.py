import numpy as np
from dataclasses import dataclass

from pypas.model.lifetime import LifetimeModel


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
