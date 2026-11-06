import math

import pytest

from scipas.model import Sample, Material, Layer


@pytest.fixture
def material() -> Material:
    return Material(name="Silicon", diffusion=1, mobility=1, bulk_annihilation_rate=1)


@pytest.fixture
def layer(material) -> Layer:
    return Layer(start=5.0, width=10.0, material=material)


def test_absorption_length_is_required(layer):
    with pytest.raises(TypeError):
        Sample(layers=[layer])


@pytest.mark.parametrize("bad_value", [0, -1.0, math.nan])
def test_non_positive_or_nan_absorption_length_raises_before_mutating_layers(layer, bad_value):
    with pytest.raises(ValueError):
        Sample(layers=[layer], absorption_length=bad_value)
    assert layer.start == 5.0


@pytest.mark.parametrize("good_value", [1e-6, 1.0, math.inf])
def test_positive_or_inf_absorption_length_constructs(layer, good_value):
    sample = Sample(layers=[layer], absorption_length=good_value)
    assert sample.absorption_length == good_value
