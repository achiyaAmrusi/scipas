import pandas as pd
from pypas.core.const import ELECTRON_REST_MASS_KEV


def compute_s_lineshape(
        db_spectra: list,
        energies,
        energy_domain_total,
        energy_domain_s,
        centralize: bool = True,
        center_value: float = ELECTRON_REST_MASS_KEV,
) -> pd.Series:
    """
    Compute S-parameter vs. positron beam energy from a series of DB spectra.

    Parameters
    ----------
    db_spectra : list of PASdb
        DB spectra, one per beam energy, in the same order as ``energies``.
    energies : array-like
        Positron beam energies [keV], one per spectrum.
    energy_domain_total : iterable of float
        (e_low, e_high) — full integration window for normalization [keV].
    energy_domain_s : iterable of float
        (e_low, e_high) — central window for the S parameter [keV].
    centralize : bool, optional
        If True (default), recenters each spectrum's axis calibration so the
        annihilation peak aligns with ``center_value`` before computing.
        Modifies each PASdb's axis calibration in place.
    center_value : float, optional
        Target energy for peak centralization [keV].
        Defaults to the electron rest mass energy (``ELECTRON_REST_MASS_KEV``).
        Use 0.0 for CDB or momentum spectra expressed as energy shifts.

    Returns
    -------
    pd.Series of uncertainties.ufloat
        S parameter vs. beam energy, indexed by ``energies``, named ``'S'``.

    Raises
    ------
    ValueError
        If ``db_spectra`` and ``energies`` have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from scispectrum import Spectrum, AxisCalibration, ResolutionCalibration
    >>> from pypas.core.db import PASdb
    >>> from pypas.analysis.vedb.lineshape import compute_s_lineshape
    >>> def make_db(center):
    ...     bins = np.linspace(center - 10, center + 10, 200)
    ...     ax = (bins[:-1] + bins[1:]) / 2
    ...     counts = np.round(1e4 * np.exp(-0.5 * ((ax - center) / 1.5) ** 2) + 50).astype(int)
    ...     spec = Spectrum(counts=counts, counts_err=np.sqrt(counts),
    ...                     axis_calib=AxisCalibration.from_array(ax),
    ...                     resolution_calib=ResolutionCalibration(lambda e: 3.0))
    ...     return PASdb.from_spectrum(spec)
    >>> db_list = [make_db(511.0), make_db(510.8)]
    >>> s = compute_s_lineshape(db_list, [5.0, 10.0],
    ...                         energy_domain_total=[507.0, 515.0],
    ...                         energy_domain_s=[510.2, 511.8])
    >>> list(s.index) == [5.0, 10.0]
    True
    >>> all(0 < float(v.nominal_value) < 1 for v in s)
    True
    """
    if len(db_spectra) != len(energies):
        raise ValueError(
            f"db_spectra and energies must have the same length "
            f"(got {len(db_spectra)} spectra and {len(energies)} energies)."
        )

    s_values = []
    for db in db_spectra:
        if centralize:
            db.recenter(center_value)
        s_values.append(db.s_parameter_calculation(energy_domain_total, energy_domain_s))

    s = pd.Series(s_values, index=list(energies), name='S')
    s.index.name = 'energy'
    return s


def compute_w_lineshape(
        db_spectra: list,
        energies,
        energy_domain_total,
        energy_domain_w_left,
        energy_domain_w_right,
        centralize: bool = True,
        center_value: float = ELECTRON_REST_MASS_KEV,
) -> pd.Series:
    """
    Compute W-parameter vs. positron beam energy from a series of DB spectra.

    Parameters
    ----------
    db_spectra : list of PASdb
        DB spectra, one per beam energy, in the same order as ``energies``.
    energies : array-like
        Positron beam energies [keV], one per spectrum.
    energy_domain_total : iterable of float
        (e_low, e_high) — full integration window for normalization [keV].
    energy_domain_w_left : iterable of float
        (e_low, e_high) — left wing window for the W parameter [keV].
    energy_domain_w_right : iterable of float
        (e_low, e_high) — right wing window for the W parameter [keV].
    centralize : bool, optional
        If True (default), recenters each spectrum's axis calibration so the
        annihilation peak aligns with ``center_value`` before computing.
        Modifies each PASdb's axis calibration in place.
    center_value : float, optional
        Target energy for peak centralization [keV].
        Defaults to the electron rest mass energy (``ELECTRON_REST_MASS_KEV``).
        Use 0.0 for CDB or momentum spectra expressed as energy shifts.

    Returns
    -------
    pd.Series of uncertainties.ufloat
        W parameter vs. beam energy, indexed by ``energies``, named ``'W'``.

    Raises
    ------
    ValueError
        If ``db_spectra`` and ``energies`` have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from scispectrum import Spectrum, AxisCalibration, ResolutionCalibration
    >>> from pypas.core.db import PASdb
    >>> from pypas.analysis.vedb.lineshape import compute_w_lineshape
    >>> def make_db(center):
    ...     bins = np.linspace(center - 10, center + 10, 200)
    ...     ax = (bins[:-1] + bins[1:]) / 2
    ...     counts = np.round(1e4 * np.exp(-0.5 * ((ax - center) / 1.5) ** 2) + 50).astype(int)
    ...     spec = Spectrum(counts=counts, counts_err=np.sqrt(counts),
    ...                     axis_calib=AxisCalibration.from_array(ax),
    ...                     resolution_calib=ResolutionCalibration(lambda e: 3.0))
    ...     return PASdb.from_spectrum(spec)
    >>> db_list = [make_db(511.0), make_db(510.8)]
    >>> w = compute_w_lineshape(db_list, [5.0, 10.0],
    ...                         energy_domain_total=[507.0, 515.0],
    ...                         energy_domain_w_left=[507.5, 509.3],
    ...                         energy_domain_w_right=[512.7, 514.5])
    >>> list(w.index) == [5.0, 10.0]
    True
    >>> all(0 < float(v.nominal_value) < 1 for v in w)
    True
    """
    if len(db_spectra) != len(energies):
        raise ValueError(
            f"db_spectra and energies must have the same length "
            f"(got {len(db_spectra)} spectra and {len(energies)} energies)."
        )

    w_values = []
    for db in db_spectra:
        if centralize:
            db.recenter(center_value)
        w_values.append(db.w_parameter_calculation(
            energy_domain_total, energy_domain_w_left, energy_domain_w_right))

    w = pd.Series(w_values, index=list(energies), name='W')
    w.index.name = 'energy'
    return w
