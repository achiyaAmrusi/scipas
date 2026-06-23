# tests/transport/test_annihilation.py
import pytest
import numpy as np
import xarray as xr
from scipas.model import Sample, Material, Layer
from scipas.analysis import compute_annihilation_fractions

@pytest.fixture
def tst_one_layer_sample()->Sample:
    silicon = Material(name="Silicon",diffusion = 1,mobility = 1,bulk_annihilation_rate = 1)
    layer = Layer(start=0.0, width=10000.0, material=silicon)
    sample = Sample(layers=[layer], absorption_length=1)
    return sample

@pytest.fixture
def tst_two_layer_sample()->Sample:
    silicon_dioxide = Material(name="Silicon dioxide",diffusion = 1,mobility = 1,bulk_annihilation_rate = 2)
    silicon = Material(name="Silicon",diffusion = 1,mobility = 1,bulk_annihilation_rate = 1)
    layer_1 = Layer(start=0.0, width=1000.0, material=silicon_dioxide)
    layer_2 = Layer(start=layer_1.width, width=9000.0, material=silicon)
    sample = Sample(layers=[layer_1, layer_2], absorption_length=np.inf)
    return sample

@pytest.fixture
def uniform_profile(tst_one_layer_sample: Sample)->xr.DataArray:
    depth = np.arange(0, tst_one_layer_sample.layers[0].width + 1, 1)
    positron_annihilation_profile = xr.DataArray(np.ones_like(depth), coords={'x': depth})
    positron_annihilation_profile /= positron_annihilation_profile.integrate('x')
    return positron_annihilation_profile

def test_warns_when_profile_exceeds_sample(tst_one_layer_sample: Sample):
    depth_long = np.arange(0, tst_one_layer_sample.sample_length() * 2, 1)
    profile_long = xr.DataArray(np.ones_like(depth_long), coords={'x': depth_long})
    profile_long /= profile_long.integrate('x')
    with pytest.warns(UserWarning, match='extends'):
        compute_annihilation_fractions(profile_long, tst_one_layer_sample)

def test_fractions_sum_to_one(tst_one_layer_sample: Sample, uniform_profile: xr.DataArray):
    res = compute_annihilation_fractions(uniform_profile, tst_one_layer_sample)
    assert np.isclose(float(res.sum()), 1.0, rtol=1e-5)

def test_surface_fraction_value(tst_one_layer_sample: Sample, uniform_profile: xr.DataArray):
    res = compute_annihilation_fractions(uniform_profile, tst_one_layer_sample)
    assert np.isclose(float(res.sel(layer=-1)), 1e-4, rtol=1e-3)

def test_two_layers_sample(tst_two_layer_sample: Sample, uniform_profile: xr.DataArray):
    res = compute_annihilation_fractions(uniform_profile, tst_two_layer_sample)
    one_layer_fraction = float(res.sel(layer=0))
    second_layer_fraction = float(res.sel(layer=1))
    layer_0 = tst_two_layer_sample.layers[0]
    layer_1 = tst_two_layer_sample.layers[1]
    annihilation_rate_ratio = layer_0.material.bulk_annihilation_rate/layer_1.material.bulk_annihilation_rate
    width_ratio = layer_0.width/layer_1.width
    assert np.isclose(one_layer_fraction + second_layer_fraction, 1.0, rtol = 1e-5)
    assert np.isclose(one_layer_fraction/second_layer_fraction, annihilation_rate_ratio* width_ratio, rtol=1e-5)
