import numpy as np
import pandas as pd
from warnings import warn
from scipy.integrate import cumulative_trapezoid
from scipy.constants import centi, micro, nano
import xarray as xr

from pyPAS.transport.implantation.utils import  get_layer_indices
from pyPAS.transport.implantation.profiles import ghosh_profile, makhov_profile

def compute_cumulative_profile(
        positron_energy: float,
        material_params: pd.Series,
        density: float,
        implantation_profile_function=ghosh_profile,
        depth_multiplier: float = 100.0,
        num_bin=10000
):
    """
    Compute the normalized cumulative implantation profile for a given material.
    The function use the z_bar to estimate how much depth it needs for the cumulative function.
    Thus, for cumulative values close to 1 the function will fail.

    Parameters
    ----------
    positron_energy : float
        Positron implantation energy in keV.
    material_params : pd.Series
        Material parameters (e.g., from gosh_material_parameters()).
    density : float
        Actual layer density (g/cm^3).
    implantation_profile_function : Callable
        Function used to compute the implantation PDF.
    depth_multiplier : float
        Factor to scale the mean depth for adequate range.

    Returns
    -------
    depth : np.ndarray
        Depth grid in nm.
    cumulative : np.ndarray
        Normalized cumulative distribution function (CDF).
    """
    if implantation_profile_function == ghosh_profile:
        z_bar = (material_params['B'] * (material_params['density'] / density)) * positron_energy ** material_params[
            'n']
    elif implantation_profile_function == makhov_profile:
        m = material_params['m']
        n = material_params['n']
        a_half = material_params['A_half'] * micro # gr/cm**2

        z_half = a_half * positron_energy ** n / density * centi/nano
        z_bar = z_half / (np.log(2)) ** (1 / m)
    else:
        raise ValueError("Unknown profile function")

    depth = np.linspace(0, z_bar * depth_multiplier, num_bin)
    profile = implantation_profile_function(positron_energy, depth, density, material_params)

    cumulative = cumulative_trapezoid(profile.values, depth, initial=0)
    cumulative /= cumulative[-1]  # normalize to 1

    return depth, cumulative


def multilayer_implantation_profile(positron_energy: float, depth_vector: np.ndarray,
                                    widths: list, materials_parameters: list, densities: list,
                                    implantation_profile_function=ghosh_profile):
    """
    Calculate the positrons implantation profile in a multilayer model using cumulative distribution of positrons in each material.
    The motivation to use this method is that approximately energetic positrons see the electrons as cloud.
    This is not correct in general and was not verified by the auther!
    It is stressed here,
    that for full analysis direct MC simulation are probably safer for implantation in complex structures.

    Returns
    -------
    pdf : xr.DataArray
        the positron implantation profile [positron/nm]

    >>> import numpy as np
    >>> from pyPAS.transport import makhov_material_parameters, makhov_profile, multilayer_implantation_profile
    >>> depth_vector = np.arange(0, 5e3, 1)
    >>> Be_parms = makhov_material_parameters().iloc[0]
    >>> pos_profile = multilayer_implantation_profile(positron_energy=10, depth_vector=depth_vector,
    ...                                               widths=[5], materials_parameters=[Be_parms],
    ...                                               densities=[Be_parms.density],
    ...                                               implantation_profile_function=makhov_profile)
    >>> np.all(pos_profile >= 0).item()
    True
    >>> pos_profile.sum().item() > 0
    True
    """

    if depth_vector[-1] > sum(widths):
        warn('The implantation depth is larger than the total model width.\n'
             'The extra depth will be computed based on the last layer.')
    if depth_vector[-1] < sum(widths[:-1]):
        warn('The implantation depth does not reach the last layer.')

    layers = get_layer_indices(depth_vector, widths)

    if sum(widths) < depth_vector[-1]:
        materials_parameters.append(materials_parameters[-1])
        densities.append(densities[-1])

    cumulatives = []
    for i, parms in enumerate(materials_parameters):
        cumulatives.append(
            compute_cumulative_profile(positron_energy=positron_energy, material_params=parms, density=densities[i],
                                       implantation_profile_function=implantation_profile_function))

    cumulative_profile = np.ones_like(depth_vector)
    cumulative_total = 0.0
    for layer in layers:
        start, end, idx = layer.ind_start, layer.ind_end, layer.layer_number
        material_depth, material_cumulative = cumulatives[idx]
        # to get smoth cumulative in the itersection between layers for idx>0 we need additional depth point
        if idx > 0:
            local_depth = depth_vector[start:end + 1] - depth_vector[start]
        else:
            local_depth = depth_vector[start:end] - depth_vector[start]

        # calculate the point at which the implantation cumulative, in the layer material alone, gets the value of cumulative_total
        z0 = np.interp(cumulative_total, material_cumulative, material_depth,
                       left=material_depth[0], right=material_depth[-1])
        # interpulate from z0 to z0+layer width
        shifted_depth = local_depth + z0
        interp_cdf = np.interp(shifted_depth, material_depth, material_cumulative, left=0, right=1)

        if idx > 0:
            # Note: in the final layer of the model, the length of interp_cdf might be shorter than expected by 1 so we us [:len(interp_cdf)-1] to regulate
            cumulative_profile[start:start + len(interp_cdf[1:])] = interp_cdf[1:]
        else:
            cumulative_profile[start:start + len(interp_cdf)] = interp_cdf
        cumulative_total = interp_cdf[-1]

    cumulative_profile = xr.DataArray(cumulative_profile, coords={'x': depth_vector})
    # differentiate in a central manner the CDF to get the PDF
    pdf = cumulative_profile.diff('x') / cumulative_profile.x.diff('x')
    pdf.coords['x'] = cumulative_profile.x[:-1] + np.diff(cumulative_profile.x.values) / 2

    return pdf