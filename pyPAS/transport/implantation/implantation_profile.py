import numpy as np
import pandas as pd
from warnings import warn
from scipy.integrate import cumulative_trapezoid
from scipy.constants import centi, micro, nano
import xarray as xr
from pyPAS.transport.implantation.utils import  get_layer_indices

def ghosh_profile(depth_vector, positron_energy, density, gosh_parms):
    """
    positron implantation profile pdf according to the ghosh profile [1].
    Parameters for profile are included in this package library via the Material package for coomon model [1,2].
    For complex material it is recommended to run MC simulation.
    Parameters
    ----------
    - depth_vector: np.ndarray
    The depth grid [nm-meters]
    - positron_energy: float
    the positron energy in keV
    - density: float [gr/cc]
    density of the material in gr/cc
    - gosh_parms: dictionary
    the parameters for the fit which include the index - l, m, clm, Nlm, n, and B [nanometer/ keV**n]
    for example, aluminum parameters can be extracted using gosh_material_parmeters().iloc[4]
    Returns
    -------
    The implanted thermalized positron distribution  [positrons/nm]
    Reference
    ---------
    [1] V.J. Ghosh et al. https://doi.org/10.1016/0169-4332(94)00331-9.
    [2] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.

    Examples
    --------
    >>> import numpy as np
    >>> from pyPAS.transport import ghosh_profile, ghosh_material_parameters
    >>> depth_vector = np.arange(0, 5, 0.01)
    >>> Be_parms = ghosh_material_parameters().iloc[0]
    >>> pos_profile = ghosh_profile(depth_vector=depth_vector, positron_energy=10,
    ...                         density=Be_parms.density, gosh_parms=Be_parms)
    >>> np.all(pos_profile >= 0).item()
    True
    >>> pos_profile.sum().item() > 0
    True
    """
    if ('l' not in gosh_parms) or ('m' not in gosh_parms) or ('N_lm' not in gosh_parms) or (
            'c_lm' not in gosh_parms) or ('B' not in gosh_parms) or ('n' not in gosh_parms) or (
            'density' not in gosh_parms):
        raise KeyError("The function requires gosh_parms to be a pd.Series from DataFrame gosh_material_parmeters()")
    l = gosh_parms['l']
    m = gosh_parms['m']
    N_lm = gosh_parms['N_lm']
    c_lm = gosh_parms['c_lm']
    z_bar = (gosh_parms['B'] * (gosh_parms['density'] / density)) * positron_energy ** gosh_parms['n']
    profile = xr.DataArray((N_lm / z_bar) * ((depth_vector / (c_lm * z_bar)) ** l) * np.exp(-(depth_vector / (c_lm * z_bar)) ** m),
        coords={'x': depth_vector})
    return profile


def makhov_profile(depth_vector, positron_energy, density, makhov_parms):
    """
    positron implantation profile pdf according to the makovian profile [1].
    The parameters the profile are included in this package library via the Material package,
    For complex material it is recommended to run MC simulation.
        Parameters
        ----------
        - depth_vector: np.ndarray # nm
        The depth grid [nm-meters]
        - positron_energy: float
        the positron energy in keV
        - density: float [gr/cc]
        density of the material in gr/cc
        - makhov_parms: dictionary
        the parameters for the fit which include the index - n, m, A_half[gr/cm**2 *keV**n]

        Returns
        -------
    The implanted thermalized positron distribution  [positrons/nano meter]

    [1] Jerzy Dryzek et al. https://doi.org/10.1016/j.nimb.2008.06.033.
        Examples
    --------
    >>> import numpy as np
    >>> from pyPAS.transport import makhov_material_parameters, makhov_profile
    >>> depth_vector = np.arange(0, 5, 0.01)
    >>> Be_parms = makhov_material_parameters().iloc[0]
    >>> pos_profile = makhov_profile(depth_vector=depth_vector, positron_energy=10,
    ...                         density=Be_parms.density, makhov_parms=Be_parms)
    >>> np.all(pos_profile >= 0).item()
    True
    >>> pos_profile.sum().item() > 0
    True
    """
    if ('m' not in makhov_parms) or ('n' not in makhov_parms) or ('A_half' not in makhov_parms):
        raise KeyError(
            "The function requires makhov_parms to be a pd.Series from DataFrame makhov_material_parameters()")

    m = makhov_parms['m']
    n = makhov_parms['n']
    a_half = makhov_parms['A_half'] * micro #  gr/ centi**2

    z_half = a_half * positron_energy ** n / density * centi / nano# nano
    z_0 = z_half / (np.log(2)) ** (1 / m)
    return xr.DataArray(m * (depth_vector ** (m - 1) / z_0 ** m) * np.exp(-(depth_vector / z_0) ** m),
                        coords={'x': depth_vector})


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
    else:
        m = material_params['m']
        n = material_params['n']
        a_half = material_params['A_half'] * micro # gr/cm**2

        z_half = a_half * positron_energy ** n / density * centi/nano
        z_bar = z_half / (np.log(2)) ** (1 / m)

    depth = np.linspace(0, z_bar * depth_multiplier, num_bin)
    profile = implantation_profile_function(depth, positron_energy, density, material_params)

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

        # calculate the point at which the cumulative of the implantation, in the layer material alone, gets the value of cumulative_total
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