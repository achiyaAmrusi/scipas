import numpy as np
import pandas as pd
import pytest
from uncertainties import UFloat

from scispectrum import Spectrum, AxisCalibration, ResolutionCalibration
from pypas.core.db import DB
from pypas.analysis.vedb.lineshape import compute_s_lineshape, compute_w_lineshape


BEAM_ENERGIES = [2.0, 5.0, 10.0, 15.0, 20.0]

ENERGY_DOMAIN_TOTAL = [507.0, 515.0]
ENERGY_DOMAIN_S     = [510.2, 511.8]
ENERGY_DOMAIN_W_L   = [507.5, 509.3]
ENERGY_DOMAIN_W_R   = [512.7, 514.5]


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_db(center: float = 511.0, sigma: float = 1.5) -> DB:
    bins = np.linspace(center - 15, center + 15, 400)
    ax = (bins[:-1] + bins[1:]) / 2
    counts = np.round(5e4 * np.exp(-0.5 * ((ax - center) / sigma) ** 2) + 200).astype(int)
    spec = Spectrum(
        counts=counts,
        counts_err=np.sqrt(counts),
        axis_calib=AxisCalibration.from_array(ax),
        resolution_calib=ResolutionCalibration(lambda e: sigma * 2 * np.sqrt(2 * np.log(2))),
    )
    return DB.from_spectrum(spec)


@pytest.fixture
def db_list():
    """Five synthetic DB objects with peaks centered at 511 keV."""
    return [_make_db(511.0) for _ in BEAM_ENERGIES]


@pytest.fixture
def db_list_off_center():
    """Five synthetic DB objects whose peaks are shifted from 511 keV."""
    return [_make_db(511.3) for _ in BEAM_ENERGIES]


# ── compute_s_lineshape ───────────────────────────────────────────────────────

def test_s_returns_series(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert isinstance(s, pd.Series)


def test_s_index_matches_energies(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert list(s.index) == BEAM_ENERGIES


def test_s_index_name(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert s.index.name == 'energy'


def test_s_series_name(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert s.name == 'S'


def test_s_values_are_ufloat(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert all(isinstance(v, UFloat) for v in s)


def test_s_values_in_range(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert all(0 < float(v.nominal_value) < 1 for v in s)


def test_s_values_have_uncertainty(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert all(v.std_dev > 0 for v in s)


def test_s_length_mismatch_raises(db_list):
    with pytest.raises(ValueError, match="same length"):
        compute_s_lineshape(db_list, BEAM_ENERGIES[:-1], ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)


def test_s_no_centralize(db_list):
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S,
                            centralize=False)
    assert all(0 < float(v.nominal_value) < 1 for v in s)


def test_s_centralize_off_center(db_list_off_center):
    """Centralizing a shifted peak should still yield sensible S values."""
    s = compute_s_lineshape(db_list_off_center, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_S, centralize=True)
    assert all(0 < float(v.nominal_value) < 1 for v in s)


# ── compute_w_lineshape ───────────────────────────────────────────────────────

def test_w_returns_series(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert isinstance(w, pd.Series)


def test_w_index_matches_energies(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert list(w.index) == BEAM_ENERGIES


def test_w_index_name(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert w.index.name == 'energy'


def test_w_series_name(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert w.name == 'W'


def test_w_values_are_ufloat(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert all(isinstance(v, UFloat) for v in w)


def test_w_values_in_range(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert all(0 < float(v.nominal_value) < 1 for v in w)


def test_w_values_have_uncertainty(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert all(v.std_dev > 0 for v in w)


def test_w_length_mismatch_raises(db_list):
    with pytest.raises(ValueError, match="same length"):
        compute_w_lineshape(db_list, BEAM_ENERGIES[:-1], ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)


def test_w_no_centralize(db_list):
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R, centralize=False)
    assert all(0 < float(v.nominal_value) < 1 for v in w)


def test_w_centralize_off_center(db_list_off_center):
    """Centralizing a shifted peak should still yield sensible W values."""
    w = compute_w_lineshape(db_list_off_center, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R, centralize=True)
    assert all(0 < float(v.nominal_value) < 1 for v in w)


# ── S and W are independent ───────────────────────────────────────────────────

def test_sw_sum_less_than_one(db_list):
    """S and W measure non-overlapping windows — their sum must be < 1."""
    s = compute_s_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    w = compute_w_lineshape(db_list, BEAM_ENERGIES, ENERGY_DOMAIN_TOTAL,
                            ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert all(float(s[e].nominal_value) + float(w[e].nominal_value) < 1.0
               for e in BEAM_ENERGIES)
