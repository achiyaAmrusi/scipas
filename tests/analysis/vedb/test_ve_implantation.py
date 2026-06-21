import numpy as np
import pytest
import xarray as xr

from pypas.transport import makhov_material_parameters, makhov_profile, ghosh_material_parameters, ghosh_profile
from pypas.analysis.vedb.ve_implanation import variable_energy_implantation_profiles

ENERGIES = [2.0, 5.0, 10.0, 20.0]
DEPTH = np.linspace(0, 10000, 2000)


@pytest.fixture
def cu_params():
    params = makhov_material_parameters()
    cu = params[params['Material'] == 'Cu'].iloc[0]
    return cu


@pytest.fixture
def si_params():
    params = makhov_material_parameters()
    return params[params['Material'] == 'Si'].iloc[0]


# ── return type and structure ─────────────────────────────────────────────────

def test_returns_list(cu_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    assert isinstance(profiles, list)


def test_length_matches_energies(cu_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    assert len(profiles) == len(ENERGIES)


def test_each_profile_is_xarray(cu_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    assert all(isinstance(p, xr.DataArray) for p in profiles)


def test_each_profile_has_x_coordinate(cu_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    assert all('x' in p.coords for p in profiles)


def test_profiles_nonnegative(cu_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    assert all(float(p.min()) >= 0 for p in profiles)


def test_profiles_normalised(cu_params):
    """Integral of each profile over depth should be ≈ 1."""
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    for p in profiles:
        integral = float(np.trapezoid(p.values, p.x.values))
        assert np.isclose(integral, 1.0, rtol=0.05), f"Profile integral = {integral:.4f}"


# ── physical behaviour ────────────────────────────────────────────────────────

def test_higher_energy_deeper_mean_depth(cu_params):
    """Mean implantation depth must increase monotonically with beam energy."""
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_params], [cu_params.density], [10000])
    mean_depths = [float((p * p.x).integrate('x')) for p in profiles]
    assert all(mean_depths[i] < mean_depths[i + 1] for i in range(len(mean_depths) - 1))


# ── input lists are not mutated ───────────────────────────────────────────────

def test_input_lists_not_mutated(cu_params):
    """multilayer_implantation_profile mutates its inputs; the wrapper must guard against this."""
    mats = [cu_params]
    dens = [cu_params.density]
    wids = [500]   # depth_vector[-1] > sum(widths) → mutation would occur inside
    with pytest.warns(UserWarning, match="implantation depth is larger"):
        variable_energy_implantation_profiles(ENERGIES, DEPTH, mats, dens, wids)
    assert len(mats) == 1
    assert len(dens) == 1
    assert len(wids) == 1


# ── multilayer case ───────────────────────────────────────────────────────────

def test_multilayer_returns_correct_length(cu_params, si_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH,
        materials_parameters=[cu_params, si_params],
        densities=[cu_params.density, si_params.density],
        widths=[500, 9500],
    )
    assert len(profiles) == len(ENERGIES)


def test_multilayer_profiles_nonnegative(cu_params, si_params):
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH,
        materials_parameters=[cu_params, si_params],
        densities=[cu_params.density, si_params.density],
        widths=[500, 9500],
    )
    assert all(float(p.min()) >= 0 for p in profiles)


# ── alternative profile function ─────────────────────────────────────────────

def test_ghosh_profile_function(cu_params):
    params = ghosh_material_parameters()
    cu_ghosh = params[params['Material'] == 'Cu'].iloc[0]
    profiles = variable_energy_implantation_profiles(
        ENERGIES, DEPTH, [cu_ghosh], [cu_ghosh.density], [10000],
        implantation_profile_function=ghosh_profile,
    )
    assert len(profiles) == len(ENERGIES)
    assert all(float(p.min()) >= 0 for p in profiles)
