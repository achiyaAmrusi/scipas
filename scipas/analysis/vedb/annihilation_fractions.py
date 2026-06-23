import numpy as np
import xarray as xr
from warnings import warn
from scipas.model.sample import Sample


def compute_annihilation_fractions(positron_profile: xr.DataArray, sample: Sample) -> xr.DataArray:
    """
    Compute the annihilation fraction of positrons in each layer and at the surface.

    The total annihilation rate is decomposed into contributions from:
        - Surface (layer = -1): positrons annihilating at the sample surface,
          governed by the surface absorption length.
        - Bulk layers (layer = 0, 1, ...): positrons annihilating in each
          material layer, weighted by the effective annihilation rate.

    The fractions are normalized so that they sum to 1.

    Parameters
    ----------
    positron_profile : xr.DataArray
        1D positron density profile after diffusion [positrons/nm],
        with a single depth coordinate (any name is accepted).
        Must be normalized such that its integral equals 1.
    sample : Sample
        Multilayer sample defining geometry, material annihilation rates,
        and surface absorption length.

    Returns
    -------
    xr.DataArray
        Normalized annihilation fractions with coordinate 'layer':
        - layer = -1  : surface annihilation fraction
        - layer = 0, 1, 2, ... : bulk annihilation fraction per layer

    Raises
    ------
    ValueError
        If positron_profile is not 1D.

    Warns
    -----
    UserWarning
        If the positron profile extends beyond the sample length.
        Annihilation beyond the sample boundary is ignored and
        fractions will not sum to 1 correctly.

     Examples
     --------
    >>> from scipas.model import Sample, Layer, Material
    >>> from scipas.analysis import compute_annihilation_fractions
    >>> import xarray as xr
    >>> import numpy as np
    >>> silicon = Material(name="Silicon",
    ...                    diffusion=1,
    ...                    mobility=1,
    ...                    bulk_annihilation_rate=1)
    >>> layer = Layer(start=0.0, width=10000.0, material=silicon)
    >>> sample = Sample(layers=[layer], absorption_length=1)
    >>> depth = np.arange(0, layer.width+1, 1)
    >>> positron_annihilation_profile = xr.DataArray(np.ones_like(depth), coords={'x':depth})
    >>> positron_annihilation_profile /= positron_annihilation_profile.integrate('x')
    >>> res = compute_annihilation_fractions(positron_annihilation_profile, sample)
    >>> round(float(res.sum())) == 1.0
    True
    >>> round(float(res.sel(layer=-1)), 5) == 1e-4
    True
    """
    # check input

    if positron_profile.ndim != 1:
        raise ValueError(f"positron_profile must be 1D, got {positron_profile.ndim}D")
    depth_dim = positron_profile.dims[0]  # infer axis name

    profile_max_depth = positron_profile.coords[depth_dim].max().item()
    sample_length = sample.sample_length()
    if profile_max_depth > sample_length:
        warn(
            f"Positron profile extends to {profile_max_depth:.1f} nm but sample ends at "
            f"{sample_length:.1f} nm. Annihilation beyond the sample boundary is ignored, "
            f"fractions will not sum to 1 correctly. Consider extending the last layer."
        )
    # defs
    layers = sample.layers
    num_of_layers = len(layers)
    layers_names = range(-1, num_of_layers)
    annihilation_rate = np.zeros(num_of_layers+1) # layers and surface

    # surface positrons annihilation rate
    annihilation_rate[0] = (positron_profile.sel({depth_dim: 0.0}, method='nearest').item() *
                             sample.layers[0].material.diffusion / sample.absorption_length)

    # layers positrons annihilation rates
    for i, layer in enumerate(layers):
        layer_positron_profile = positron_profile.sel({depth_dim: slice(layer.start, layer.start + layer.width)})
        positron_fraction_in_layer = layer_positron_profile.integrate(depth_dim)
        annihilation_rate[i+1] = layer.material.effective_annihilation_rate() * positron_fraction_in_layer.item()

    # norm
    annihilation_fractions = xr.DataArray(annihilation_rate, coords={'layer':layers_names})/annihilation_rate.sum()

    return annihilation_fractions



