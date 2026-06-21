import numpy as np
import xarray as xr
from pypas.transport.implantation import multilayer_implantation_profile, makhov_profile


def variable_energy_implantation_profiles(
        energies,
        depth_vector: np.ndarray,
        materials_parameters: list,
        densities: list,
        widths: list,
        implantation_profile_function=makhov_profile,
) -> list:
    """
    Compute positron implantation profiles for a series of beam energies.

    Wraps ``multilayer_implantation_profile`` in a loop over ``energies``,
    returning one profile per energy. Handles both single-layer (homogeneous
    substrate) and multilayer (thin film on substrate) geometries.

    Parameters
    ----------
    energies : array-like
        Positron beam energies [keV] for which to compute implantation profiles.
    depth_vector : np.ndarray
        1D depth grid [nm] spanning the full sample, e.g.
        ``np.linspace(0, sample.sample_length(), 10000)``.
    materials_parameters : list of pd.Series
        Material parameters for each layer, one entry per layer.
        Obtain via ``makhov_material_parameters()`` or ``ghosh_material_parameters()``.
        For a homogeneous substrate pass a single-element list.
    densities : list of float
        Actual material density [g/cm³] for each layer.
        Must be the same length as ``materials_parameters``.
    widths : list of float
        Width of each layer [nm], from surface to substrate.
        If ``depth_vector[-1]`` exceeds ``sum(widths)``, the last layer is
        extended to fill the remaining depth (semi-infinite substrate approximation).
        For a homogeneous substrate pass ``[sample_length]``.
    implantation_profile_function : callable, optional
        Profile function to use for each layer.
        Must have the signature
        ``f(positron_energy, depth_vector, density, material_params) -> xr.DataArray``.
        Defaults to ``makhov_profile``.

    Returns
    -------
    list of xr.DataArray
        Implantation profiles, one per entry in ``energies``, each with
        coordinate ``'x'`` in nm and units of positrons/nm (normalised so
        the integral over depth equals 1).

    Examples
    --------
    >>> import numpy as np
    >>> from pypas.transport import makhov_material_parameters, makhov_profile
    >>> from pypas.analysis.vedb.ve_implanation import variable_energy_implantation_profiles
    >>> params = makhov_material_parameters()
    >>> cu = params[params['Material'] == 'Cu'].iloc[0]
    >>> depth = np.linspace(0, 10000, 5000)
    >>> profiles = variable_energy_implantation_profiles(
    ...     energies=[5.0, 10.0, 20.0],
    ...     depth_vector=depth,
    ...     materials_parameters=[cu],
    ...     densities=[cu.density],
    ...     widths=[10000],
    ... )
    >>> len(profiles)
    3
    >>> all(isinstance(p, xr.DataArray) for p in profiles)
    True
    >>> all('x' in p.coords for p in profiles)
    True
    """
    profiles = []
    for energy in energies:
        profile = multilayer_implantation_profile(
            positron_energy=float(energy),
            depth_vector=depth_vector,
            widths=list(widths),
            materials_parameters=list(materials_parameters),
            densities=list(densities),
            implantation_profile_function=implantation_profile_function,
        )
        profiles.append(profile)
    return profiles
