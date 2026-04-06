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
                                                             resolution=resolution,
                                                             background_fraction=0).lifetime
    return response_mat.T
