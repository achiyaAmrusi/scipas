import numpy as np
import xarray as xr
from pyPAS.model.sample import Sample


def annihilation_fraction_per_layer(positron_profile: xr.DataArray, sample: Sample) -> xr.DataArray:
    """
    Calculate the  annihilation rate fraction of positron per layer.
    Parameters
    ----------
    - positron_profile: xr.DataArray
     The positron profile (in the model (after diffusion)
    -  model: Sample
     The model in which they annihilate

     Return
     -------
     xarray
     annihilation rate in each layer and surface
    """
    layers = sample.layers
    num_of_layers = len(layers)
    layers_names = [f'layer_{i}' for i in range(num_of_layers)]

    annihilation_rate = np.zeros(num_of_layers+1) # layers and surface

    # layers positrons annihilation rates
    for i, layer in enumerate(layers):
        layer_positron_profile = positron_profile.sel(x=slice(layer.start, layer.start+layer.width))
        positron_fraction_in_layer = layer_positron_profile.integrate('x')
        annihilation_rate[i] = layer.material.effective_annihilation_rate() * positron_fraction_in_layer.item()

    # surface positrons annihilation rate
    annihilation_rate[-1] = positron_profile[0].item() * sample.layers[0].material.diffusion / sample.absorbtion_length
    layers_names.append('surface')

    annihilation_fractions = xr.DataArray(annihilation_rate, coords={'layer':layers_names})/annihilation_rate.sum()

    return annihilation_fractions



