import numpy as np
from scipy.constants import centi, micro, nano
import xarray as xr

def ghosh_profile(positron_energy,depth_vector,  density, gosh_parms):
    """
    positron implantation profile pdf according to the ghosh profile [1].
    Parameters for profile are included in this package library via the Material package for coomon model [1,2].
    For complex material it is recommended to run MC simulation.
    Parameters
    ----------
    - positron_energy: float
    the positron energy in keV
    - depth_vector: np.ndarray
    The depth grid [nm-meters]
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
    >>> depth_vector = np.arange(0, 5e3, 1)
    >>> Be_parms = ghosh_material_parameters().iloc[0]
    >>> pos_profile = ghosh_profile(positron_energy=10, depth_vector=depth_vector,
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


def makhov_profile( positron_energy, depth_vector, density, makhov_parms):
    """
    positron implantation profile pdf according to the makovian profile [1].
    The parameters the profile are included in this package library via the Material package,
    For complex material it is recommended to run MC simulation.
        Parameters
        ----------
        - positron_energy: float
        the positron energy in keV
        - depth_vector: np.ndarray # nm
        The depth grid [nm-meters]
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
    >>> depth_vector = np.arange(0, 5e3, 1)
    >>> Be_parms = makhov_material_parameters().iloc[0]
    >>> pos_profile = makhov_profile(positron_energy=10, depth_vector=depth_vector,
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

