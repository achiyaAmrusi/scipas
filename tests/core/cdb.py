import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyspectrum.core import Domain
from pyspectrum.domain_analysis.single_peak import center_estimator
from uncertainties import nominal_value

from pyPAS.core import PAScdb, PASdb
from pyPAS.core.const import ELECTRON_REST_MASS_KEV


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def symmetric_pairs():
    """Synthetic CDB pairs: Gaussian Doppler spread + small resolution jitter,
    both centered at 511 keV so the coincidence map peaks near (0, 0)."""
    rng = np.random.default_rng(42)
    n = 10_000
    doppler = rng.normal(0, 0.5, n)      # momentum spread → (E1-E2)/2
    resolution = rng.normal(0, 0.3, n)   # energy jitter  → (E1+E2-1022)/2
    e1 = ELECTRON_REST_MASS_KEV + doppler + resolution
    e2 = ELECTRON_REST_MASS_KEV - doppler + resolution
    return pd.DataFrame({'energy_1': e1, 'energy_2': e2})


@pytest.fixture
def cdb(symmetric_pairs):
    return PAScdb(symmetric_pairs, energy_min=-4, energy_max=4, mesh_interval=0.1)


ENERGY_MIN = -4.0
ENERGY_MAX = 4.0
MESH = 0.1


# ── construction ──────────────────────────────────────────────────────────────

def test_construction_valid(symmetric_pairs):
    cdb = PAScdb(symmetric_pairs, energy_min=ENERGY_MIN, energy_max=ENERGY_MAX, mesh_interval=MESH)
    assert isinstance(cdb, PAScdb)


def test_construction_wrong_column_names():
    df = pd.DataFrame({'E1': [511.0], 'E2': [511.0]})
    with pytest.raises(ValueError, match="energy_1"):
        PAScdb(df, energy_min=-4, energy_max=4, mesh_interval=0.1)


def test_construction_wrong_column_order():
    """Columns present but swapped — constructor must reject them."""
    df = pd.DataFrame({'energy_2': [511.0], 'energy_1': [511.0]})
    with pytest.raises(ValueError, match="energy_1"):
        PAScdb(df, energy_min=-4, energy_max=4, mesh_interval=0.1)


def test_construction_energy_min_equals_max():
    df = pd.DataFrame({'energy_1': [511.0], 'energy_2': [511.0]})
    with pytest.raises(ValueError):
        PAScdb(df, energy_min=4, energy_max=4, mesh_interval=0.1)


def test_construction_energy_min_greater_than_max():
    df = pd.DataFrame({'energy_1': [511.0], 'energy_2': [511.0]})
    with pytest.raises(ValueError):
        PAScdb(df, energy_min=5, energy_max=4, mesh_interval=0.1)


# ── coincidence_map ───────────────────────────────────────────────────────────

def test_coincidence_map_is_xarray(cdb):
    assert isinstance(cdb.coincidence_map, xr.DataArray)


def test_coincidence_map_dims(cdb):
    assert set(cdb.coincidence_map.dims) == {'resolution', 'doppler'}


def test_coincidence_map_nonnegative(cdb):
    assert (cdb.coincidence_map.values >= 0).all()


def test_coincidence_map_coordinates_within_range(cdb):
    coords = cdb.coincidence_map.coords
    assert float(coords['doppler'].min()) >= ENERGY_MIN
    assert float(coords['doppler'].max()) <= ENERGY_MAX
    assert float(coords['resolution'].min()) >= ENERGY_MIN
    assert float(coords['resolution'].max()) <= ENERGY_MAX


def test_coincidence_map_peak_near_zero(cdb):
    """For symmetric input, the Doppler peak bin should be within 0.5 keV of zero."""
    cm = cdb.coincidence_map
    doppler_marginal = cm.sum('resolution')
    doppler_peak_idx = int(np.argmax(doppler_marginal.values))
    doppler_peak = float(cm.coords['doppler'][doppler_peak_idx])
    assert abs(doppler_peak) < 0.5


def test_coincidence_map_cached(cdb):
    """Property should return the same object on repeated access."""
    assert cdb.coincidence_map is cdb.coincidence_map


# ── doppler_broadening() ──────────────────────────────────────────────────────

def test_doppler_broadening_returns_pasdb(cdb):
    assert isinstance(cdb.doppler_broadening(), PASdb)


def test_doppler_broadening_centered_at_zero(cdb):
    """Default center_value=0: after centralization, peak centroid must be near 0."""
    db = cdb.doppler_broadening(centralize_peak=True)
    center = nominal_value(center_estimator(db))
    assert abs(center) < 0.2


def test_doppler_broadening_no_centralize_returns_pasdb(cdb):
    db = cdb.doppler_broadening(centralize_peak=False)
    assert isinstance(db, PASdb)


def test_doppler_broadening_counts_nonnegative(cdb):
    db = cdb.doppler_broadening()
    assert (db.data.values >= 0).all()


def test_doppler_broadening_conserves_total_counts(cdb):
    """Projecting over the resolution axis must preserve total event count."""
    db = cdb.doppler_broadening(centralize_peak=False)
    expected = int(cdb.coincidence_map.values.sum())
    actual = int(db.data.values.sum())
    assert actual == expected


# ── resolution() ─────────────────────────────────────────────────────────────

def test_resolution_returns_domain(cdb):
    assert isinstance(cdb.resolution(), Domain)


def test_resolution_counts_nonnegative(cdb):
    res = cdb.resolution()
    assert (res.data.values >= 0).all()


def test_resolution_conserves_total_counts(cdb):
    """Projecting over the Doppler axis must preserve total event count."""
    res = cdb.resolution()
    expected = int(cdb.coincidence_map.values.sum())
    actual = int(res.data.values.sum())
    assert actual == expected


def test_resolution_peak_near_zero(cdb):
    """For symmetric input, resolution centroid should be near 0."""
    res = cdb.resolution()
    center = nominal_value(center_estimator(res))
    assert abs(center) < 0.5


# ── S/W parameters from CDB projection ───────────────────────────────────────

def test_sw_from_cdb_projection_valid_range(cdb):
    """DB projected from CDB must yield physically valid S and W parameters."""
    db = cdb.doppler_broadening(center_value=0)
    s = db.s_parameter_calculation([-3.3, 3.3], [-0.8, 0.8])
    w = db.w_parameter_calculation([-3.3, 3.3], [-3.3, -2.0], [2.0, 3.3])
    assert 0 < float(s.nominal_value) < 1
    assert 0 < float(w.nominal_value) < 1


def test_sw_sum_less_than_one(cdb):
    """S and W measure non-overlapping regions — their sum must be < 1."""
    db = cdb.doppler_broadening(center_value=0)
    s = db.s_parameter_calculation([-3.3, 3.3], [-0.8, 0.8])
    w = db.w_parameter_calculation([-3.3, 3.3], [-3.3, -2.0], [2.0, 3.3])
    assert float(s.nominal_value) + float(w.nominal_value) < 1.0


def test_s_has_uncertainty(cdb):
    db = cdb.doppler_broadening(center_value=0)
    s = db.s_parameter_calculation([-3.3, 3.3], [-0.8, 0.8])
    assert s.std_dev > 0
