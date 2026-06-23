import numpy as np
import xarray as xr

from scipas.core.lifetime import TimeResolution, PASLifetime
from scipas.model.lifetime import LifetimeModel
from scipas.analysis.lifetime.generator import generate_analytical_lt_spectrum

def _response_matrix(tau_grid: np.ndarray, time_grid: np.ndarray, resolution: TimeResolution):
    response_mat = np.zeros((len(tau_grid), len(time_grid)))

    for j, tau in enumerate(tau_grid):
        impulse_model = LifetimeModel(name='',
                                      lifetimes=np.array([tau]),
                                      intensities=[1])
        response_mat[j, :] = generate_analytical_lt_spectrum(time_grid,
                                                             model=impulse_model,
                                                             resolution=resolution).lifetime.counts
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


def t0_scan(inverter, pals: PASLifetime, t0_values: np.ndarray,
            **invert_kwargs) -> dict:
    """
    Scan t0_shift values and pick the one that gives the best inversion.

    For each t0, runs the inverter and computes the chi-squared of the
    forward-model reconstruction against the data. Returns the best t0
    and all results.

    Parameters
    ----------
    inverter : LifetimeInvert
        An initialized inversion object (TikhonovRegularization or
        MaximalEntropyInversion).
    pals : PASLifetime
        Measured lifetime spectrum.
    t0_values : np.ndarray
        Array of t0_shift values to scan.
    **invert_kwargs
        Additional keyword arguments passed to inverter.invert().

    Returns
    -------
    dict with keys:
        best_t0 : float — t0_shift with lowest chi-squared.
        best_result : tuple — inversion result at best_t0.
        chi_squared : xr.DataArray — chi-squared vs t0.
    """
    bg_est = invert_kwargs.get("bg_est", 0.0)
    counts = pals.lifetime.counts
    net_counts = counts - bg_est
    norm = np.trapezoid(net_counts, pals.lifetime.energy)
    data = net_counts / norm
    data_err = np.sqrt(np.maximum(counts, 1)) / norm

    dtau = inverter.characteristic_time_grid[1] - inverter.characteristic_time_grid[0]
    w = np.ones_like(inverter.characteristic_time_grid) * dtau
    w[0] *= 0.5
    w[-1] *= 0.5

    chi2_values = np.empty(len(t0_values))
    results = []

    for i, t0 in enumerate(t0_values):
        result = inverter.invert(pals, t0_shift=t0, **invert_kwargs)
        results.append(result)

        q = result[0] if isinstance(result[0], np.ndarray) else result[1]
        time_values = pals.lifetime.energy.values - t0
        response = _response_matrix(
            inverter.characteristic_time_grid, time_values, pals.resolution
        )
        predicted = (response * w[None, :]) @ q
        chi2_values[i] = np.sum((data - predicted) ** 2 / data_err ** 2) / len(data)

    best_idx = np.argmin(chi2_values)
    chi2_da = xr.DataArray(chi2_values, coords={"t0": t0_values}, dims=["t0"])

    return {
        "best_t0": t0_values[best_idx],
        "best_result": results[best_idx],
        "chi_squared": chi2_da,
    }