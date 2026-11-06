import pytest
from scipas.model import Sample, Material, Layer


@pytest.fixture
def material() -> Material:
    return Material(name="Silicon", diffusion=1, mobility=1, bulk_annihilation_rate=1)


def test_conflicting_explicit_start_raises(material):
    first = Layer(width=10, material=material)
    second = Layer(start=999, width=5, material=material)
    with pytest.raises(ValueError, match=r"1.*10\.0.*999"):
        Sample(layers=[first, second])


def test_omitted_start_is_auto_filled(material):
    first = Layer(width=10, material=material)
    second = Layer(width=5, material=material)
    Sample(layers=[first, second])
    assert second.start == 10.0


def test_matching_explicit_start_preserved(material):
    first = Layer(start=0.0, width=10, material=material)
    second = Layer(start=10.0, width=5, material=material)
    Sample(layers=[first, second])
    assert first.start == 0.0
    assert second.start == 10.0


def test_single_layer_conflicting_start_raises(material):
    layer = Layer(start=5.0, width=10, material=material)
    with pytest.raises(ValueError, match=r"0.*0\.0.*5\.0"):
        Sample(layers=[layer])


def test_float_drift_within_tolerance_accepted(material):
    first = Layer(width=0.1, material=material)
    second = Layer(width=0.2, material=material)
    third = Layer(start=0.3, width=1.0, material=material)
    Sample(layers=[first, second, third])
    assert third.start == 0.3


def test_reused_layer_at_different_position_raises(material):
    first = Layer(width=10, material=material)
    layer = Layer(width=5, material=material)
    Sample(layers=[first, layer])
    assert layer.start == 10.0

    other_first = Layer(width=20, material=material)
    with pytest.raises(ValueError):
        Sample(layers=[other_first, layer])
