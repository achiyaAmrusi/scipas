import numpy as np
import pytest

from pypas.transport import (
    makhov_material_parameters, makhov_profile,
    multilayer_implantation_profile,
)

DEPTH = np.linspace(0, 10000, 2000)


@pytest.fixture
def cu_params():
    params = makhov_material_parameters()
    return params[params['Material'] == 'Cu'].iloc[0]


@pytest.fixture
def si_params():
    params = makhov_material_parameters()
    return params[params['Material'] == 'Si'].iloc[0]


def test_warns_when_depth_exceeds_model_width(cu_params):
    with pytest.warns(UserWarning, match="implantation depth is larger"):
        multilayer_implantation_profile(
            10, DEPTH, [500], [cu_params], [cu_params.density],
            implantation_profile_function=makhov_profile)


def test_input_lists_mutated_by_multilayer(cu_params):
    """multilayer_implantation_profile appends to inputs when depth > model width."""
    mats = [cu_params]
    dens = [cu_params.density]
    wids = [500]
    with pytest.warns(UserWarning, match="implantation depth is larger"):
        multilayer_implantation_profile(
            10, DEPTH, wids, mats, dens,
            implantation_profile_function=makhov_profile)
    assert len(mats) == 2, "multilayer_implantation_profile should mutate input lists"
    assert len(dens) == 2


def test_profile_nonnegative(cu_params, si_params):
    profile = multilayer_implantation_profile(
        10, DEPTH, [500, 9500], [cu_params, si_params],
        [cu_params.density, si_params.density],
        implantation_profile_function=makhov_profile)
    assert float(profile.min()) >= 0


def test_profile_normalised(cu_params):
    profile = multilayer_implantation_profile(
        10, DEPTH, [10000], [cu_params], [cu_params.density],
        implantation_profile_function=makhov_profile)
    integral = float(np.trapezoid(profile.values, profile.x.values))
    assert np.isclose(integral, 1.0, rtol=0.05)
