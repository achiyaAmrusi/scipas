import numpy as np

from pyPAS.core.lt import TimeResolution
from pyPAS.lifetime.model import LifetimeModel
from pyPAS.lifetime.generator import generate_analytical_lt_spectrum

def _response_matrix(tau_grid: np.ndarray, time_grid: np.ndarray, resolution: TimeResolution):
    response_mat = np.zeros((len(tau_grid), len(time_grid)))

    for j, tau in enumerate(tau_grid):
        impulse_model = LifetimeModel(name='',
                                      lifetimes=np.array([tau]),
                                      intensities=[1])
        response_mat[j, :] = generate_analytical_lt_spectrum(time_grid,
                                                             model=impulse_model,
                                                             resolution=resolution).lifetime
    return response_mat.T

def _svd_truncate(response: np.ndarray, noise_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate SVD of response matrix, discarding singular values below noise.

    Parameters
    ----------
    response : np.ndarray
        Response matrix of shape (n_time, n_tau), already scaled by dτ.
    noise_level : float
        relative threshold for singular value truncation. Singular values
        below this value are discarded. Should be estimated from the data,
        e.g. np.mean(normlized_pals_err).

    Returns
    -------
    U, s, Vt : np.ndarray
        Truncated SVD components, shape (n_time, k), (k,), (k, n_tau).
    """
    U, s, Vt = np.linalg.svd(response, full_matrices=False)
    keep = s > s.max() * noise_level
    return U[:, keep], s[keep], Vt[keep, :]