import numpy as np
import pandas as pd
from pyspectrum import Peak
from uncertainties import ufloat
from pyPAS.pas_db import PASdb
import xarray as xr
ELECTRON_REST_MASS = 511


class PAScdb:
    """
    Coincidence Doppler Broadening (CDB) analysis class.

    This class processes pairs of coincident photon energies from CDB experiments and computes:
    - A 2D histogram representing the coincidence map in resolution and Doppler space.
    - A 1D Doppler broadening spectrum (DB) by projecting the coincidence map.
    - A 1D resolution spectrum (RES) by summing along the Doppler axis.

    ToDo: fix the background estimation or remove it from PySpectrum
    Attributes
    ----------
    pair_df : pd.DataFrame
        DataFrame containing coincidence photon energy pairs, with two columns:
        ['energy_1', 'energy_2']. Each row represents a detected event.

    Methods
    -------
    __init__(gamma_energy_pair)
        Initialize the class with a DataFrame containing with columns:
        ['energy_1', 'energy_2']. of the photon pair

    coincidence_map(energy_dynamic_range, mesh_interval) -> xr.DataArray
        Compute a 2D histogram of the coincidence data with axes:
        - x-axis: resolution = (E1 + E2 − 2·511 keV)
        - y-axis: Doppler = (E1 − E2)/2

    doppler_broadening_spectrum(energy_dynamic_range, mesh_interval) -> PASdb
        Project the 2D histogram along the resolution axis to produce a 1D Doppler spectrum.
        Returns a `PASdb` object for further analysis (e.g., S/W parameters).

    resolution_spectrum(energy_dynamic_range, mesh_interval) -> Peak
        Project the 2D histogram along the Doppler axis to produce a 1D resolution spectrum.
        Returns a `Peak` object.

    from_dataframe(df: pd.DataFrame) -> PAScdb
        Alternate constructor to create a PAScdb instance from a DataFrame with columns:
        ['energy_1', 'energy_2'].

    Notes
    -----
    - All energy ranges provided are interpreted **relative to the 511 keV** positron rest mass energy.
    - Coincidence data outside the defined dynamic range is excluded from histograms.
    - Returned spectra use dummy background estimates (ufloat(0,1)).
    """
    def __init__(self, gamma_energy_pair):
        """
        Create a PAScdb instance from a DataFrame with columns ['energy_1', 'energy_2'].

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing the measured coincidence events.
            Must contain **only** the two columns: 'energy_1' and 'energy_2'.

        Returns
        -------
        PAScdb
            A new instance of PAScdb initialized with the provided data.

        Raises
        ------
        ValueError
            If the input DataFrame does not contain the expected columns in the correct order.
        """
        expected_columns = ['energy_1', 'energy_2']
        if list(gamma_energy_pair.columns) != expected_columns:
            raise ValueError(
                f"Input DataFrame must contain only these columns in order: {expected_columns}. Got: {df.columns.tolist()}")
        self.pair_df = gamma_energy_pair

    def coincidence_map(self, energy_dynamic_range, mesh_interval):
        """
        Compute the 2D coincidence histogram (CDB) from all energy pair measurements.

        The x-axis corresponds to Doppler broadening: (E₁ − E₂)/2
        The y-axis corresponds to resolution: (E₁ + E₂ − 2·511 keV)

        Parameters
        ----------
        energy_dynamic_range : list or tuple of two floats
            The minimum and maximum energy range for both axes, **relative to 511 keV**, in keV.
            Only events whose Doppler/resolution values fall within this range will be counted.
            For example: [-4, 4] includes ±4 keV around the annihilation energy.

        mesh_interval : float
            The bin width (in keV) for the histogram axes.

        Returns
        -------
        xr.DataArray
            2D histogram of coincidences as a labeled DataArray:
            - dims: ["resolution", "doppler"]
            - coords: energy midpoints for each axis bin
            - data: event counts

        Notes
        -----
        Events with Doppler or resolution values outside the `energy_dynamic_range` are excluded.
        To include broader events (e.g., from Compton scattering or core electron annihilation), expand the dynamic range accordingly.
        """
        cdb_pairs = self.pair_df

        # Compute Doppler and resolution components
        db = (cdb_pairs['energy_1'] - cdb_pairs['energy_2']) / 2
        res = ((cdb_pairs['energy_1'] + cdb_pairs['energy_2']) - 2 * ELECTRON_REST_MASS)/2

        # Define bin edges
        e_min, e_max = energy_dynamic_range
        bin_edges_x = np.arange(e_min, e_max + mesh_interval, mesh_interval)
        bin_edges_y = np.arange(e_min, e_max + mesh_interval, mesh_interval)

        # Build 2D histogram
        coincidence_hist, x_edges, y_edges = np.histogram2d(
            res, db, bins=[bin_edges_x, bin_edges_y]
        )

        # Wrap as xarray DataArray
        coincidence_hist = xr.DataArray(
            coincidence_hist,
            coords={
                "resolution": (x_edges[:-1] + x_edges[1:]) / 2,
                "doppler": (y_edges[:-1] + y_edges[1:]) / 2,
            },
            dims=["resolution", "doppler"],
        )
        return coincidence_hist

    def doppler_broadening_spectrum(self, energy_dynamic_range, mesh_interval):
        """
        Compute the Doppler broadening spectrum (1D) from the 2D histogram.

        This is done by summing over the resolution axis, collapsing the 2D histogram
        to a function of Doppler shift only.

        Parameters
        ----------
        energy_dynamic_range : list
            Energy window for the histogram, relative to 511 keV.
            For example: [-4, 4].

        mesh_interval : float
            Bin width (in keV) along both axes.

        Returns
        -------
        PASdb
            A PASdb spectrum object containing the Doppler spectrum and dummy timing info.
        """
        coincidence_hist = self.coincidence_map(energy_dynamic_range, mesh_interval)
        doppler_broadening = coincidence_hist.sum("resolution")
        return PASdb(doppler_broadening, ufloat(0,1), ufloat(0,1))

    def resolution_spectrum(self, energy_dynamic_range, mesh_interval):
        """
        Compute the resolution spectrum (1D) from the 2D histogram.

        This is done by summing over the Doppler axis, collapsing the 2D histogram
        to a function of total energy shift only.
        Parameters
        ----------
        energy_dynamic_range : list
            Energy window for the histogram, relative to 511 keV.
            For example: [-4, 4].

        mesh_interval : float
            Bin width (in keV) along both axes.

        Returns
        -------
        Peak
            A Peak spectrum object containing the resolution spectrum and dummy timing info.
        """
        coincidence_hist = self.coincidence_map(energy_dynamic_range, mesh_interval)
        resolution = coincidence_hist.sum("doppler")
        return Peak(resolution, ufloat(0,1), ufloat(0,1))