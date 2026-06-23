import numpy as np
import pytest
from scispectrum import Spectrum, AxisCalibration, ResolutionCalibration
from scipas.core.db import DB


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gaussian_spectrum():
    """Synthetic Gaussian peak centered at 511 keV with known properties."""
    bins = np.linspace(511 - 100, 511 + 100, 1000)
    centers = (bins[1:] + bins[:-1]) / 2
    sigma = 2.0  # keV
    counts = 1e6 * np.exp(-0.5 * ((centers - 511) / sigma) ** 2) + 100  # peak + flat bg
    counts = np.round(counts).astype(int)
    resolution = ResolutionCalibration(lambda e: sigma * 2 * np.sqrt(2 * np.log(2)))
    spec = Spectrum(
        counts=counts,
        counts_err=np.sqrt(counts),
        axis_calib=AxisCalibration.from_array(centers),
        resolution_calib=resolution
    )
    return spec


@pytest.fixture
def db(gaussian_spectrum):
    return DB.from_spectrum(gaussian_spectrum)


# window definitions — mirrors real usage
# W outer edges are ~0.5 keV inside the total domain on each side (gap >> one bin ≈ 0.2 keV)
ENERGY_DOMAIN_TOTAL = [507.7, 514.9]
ENERGY_DOMAIN_S     = [510.2, 511.8]
ENERGY_DOMAIN_W_L   = [508.2, 509.3]
ENERGY_DOMAIN_W_R   = [512.7, 514.4]


# ── construction ──────────────────────────────────────────────────────────────

def test_from_spectrum_returns_pasdb(gaussian_spectrum):
    db = DB.from_spectrum(gaussian_spectrum)
    assert isinstance(db, DB)


def test_peak_centered_at_511(db):
    """After centralization, peak center should be close to 511 keV."""
    from scispectrum.domain_analysis.single_peak import center_estimator
    from uncertainties import nominal_value
    center = nominal_value(center_estimator(db))
    assert np.isclose(center, 511.0, atol=0.1)


def test_no_centralize(gaussian_spectrum):
    """With centralize_peak=False, axis should not be shifted."""
    db_raw = DB.from_spectrum(gaussian_spectrum, centralize_peak=False)
    assert isinstance(db_raw, DB)


# ── S parameter ───────────────────────────────────────────────────────────────

def test_s_parameter_between_0_and_1(db):
    s = db.s_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert 0 < float(s.nominal_value) < 1


def test_s_parameter_symmetric_peak(db):
    """For a symmetric Gaussian, S should be close to the fraction of
    a Gaussian within ±0.8 keV of center — roughly 0.3 for sigma=2."""
    s = db.s_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert 0.2 < float(s.nominal_value) < 0.5


def test_s_parameter_has_uncertainty(db):
    s = db.s_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    assert s.std_dev > 0


# ── W parameter ───────────────────────────────────────────────────────────────

def test_w_parameter_between_0_and_1(db):
    w = db.w_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert 0 < float(w.nominal_value) < 1


def test_w_parameter_has_uncertainty(db):
    w = db.w_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert w.std_dev > 0


def test_sw_sum_less_than_one(db):
    """S and W measure non-overlapping regions — their sum must be < 1."""
    s = db.s_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_S)
    w = db.w_parameter_calculation(ENERGY_DOMAIN_TOTAL, ENERGY_DOMAIN_W_L, ENERGY_DOMAIN_W_R)
    assert float(s.nominal_value) + float(w.nominal_value) < 1.0


# ── boundary validation ───────────────────────────────────────────────────────

def test_w_raises_on_invalid_boundaries(db):
    """W parameter should raise if boundaries are in wrong order."""
    with pytest.raises(ValueError):
        db.w_parameter_calculation(
            ENERGY_DOMAIN_TOTAL,
            energy_domain_w_left=[506.0, 509.3],   # left edge outside total domain
            energy_domain_w_right=ENERGY_DOMAIN_W_R
        )


def test_w_warns_on_subbin_gap(db):
    """A W window whose outer edge is within one bin of the domain edge should warn, not raise."""
    de = float(db.spectrum.axis[1] - db.spectrum.axis[0])
    total = [507.7, 514.9]
    # place left W edge half a bin inside the domain edge — guaranteed sub-bin gap
    left_w_start = total[0] + 0.5 * de
    with pytest.warns(UserWarning, match="smaller than one bin"):
        db.w_parameter_calculation(
            total,
            energy_domain_w_left=[left_w_start, 509.3],
            energy_domain_w_right=[512.7, 514.9]
        )