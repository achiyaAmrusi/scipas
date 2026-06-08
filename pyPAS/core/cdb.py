import numpy as np
import pandas as pd
from scispectrum.core import Spectrum
from scispectrum.calibration import AxisCalibration
import xarray as xr
from pyPAS.core import PASdb
from pyPAS.core.const import ELECTRON_REST_MASS_KEV

class PAScdb:
    """
    Coincidence Doppler Broadening (CDB) analysis class.

    This class processes pairs of coincident photon energies from CDB experiments and computes:
    - A 2D histogram representing the coincidence map in resolution and Doppler space.
    - A 1D Doppler broadening spectrum (DB) by projecting the coincidence map.
    - A 1D resolution spectrum (RES) by summing along the Doppler axis.

    Attributes
    ----------
    pair_df : pd.DataFrame
        DataFrame containing coincidence photon energy pairs, with two columns:
        ['energy_1', 'energy_2']. Each row represents a detected event.
    coincidence_map : xr.DataArray
        Precomputed 2D coincidence histogram (resolution × doppler), cached at
        construction. Dims: ``["resolution", "doppler"]``; coords: bin midpoints in keV.

    Methods
    -------
    doppler_broadening(centralize_peak, center_value) -> PASdb
        Project the coincidence map along the resolution axis to produce a 1D
        Doppler broadening spectrum wrapped in a ``PASdb`` domain.
    resolution() -> Domain
        Project the coincidence map along the Doppler axis to produce a 1D
        resolution (sum-energy) spectrum.

    Notes
    -----
    - All energy ranges provided are interpreted **relative to the 511 keV** positron rest mass energy.
    - Coincidence data outside the defined dynamic range is excluded from histograms.
    - Returned spectra use dummy background estimates (ufloat(0,1)).
    """

    def __init__(self, gamma_energy_pair: pd.DataFrame,
                 energy_min: float,
                 energy_max: float,
                 mesh_interval: float):
        """
        Initialize a PAScdb instance and precompute the 2D coincidence histogram.

        The coincidence map is computed once at construction from the provided
        energy bounds and bin width, and cached as ``self.coincidence_map``.
        All subsequent analysis methods (``doppler_broadening``, ``resolution``)
        project this cached map, avoiding redundant recomputation.

        Parameters
        ----------
        gamma_energy_pair : pd.DataFrame
            DataFrame containing the measured coincidence events.
            Must contain exactly two columns in order: 'energy_1' and 'energy_2',
            where each row represents a detected photon pair in keV.
        energy_min : float
            Lower bound of the energy window, relative to 511 keV, in keV.
            For example: -4.0 includes events down to 507 keV.
        energy_max : float
            Upper bound of the energy window, relative to 511 keV, in keV.
            For example: 4.0 includes events up to 515 keV.
        mesh_interval : float
            Bin width in keV for both axes of the coincidence histogram.
            Should be chosen relative to the detector energy resolution.

        Raises
        ------
        ValueError
            If ``gamma_energy_pair`` does not contain exactly the columns
            ['energy_1', 'energy_2'] in that order.
        ValueError
            If ``energy_min >= energy_max``.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'energy_1': np.random.normal(511, 1, 1000),
        ...     'energy_2': np.random.normal(511, 1, 1000)})
        >>> cdb = PAScdb(df, energy_min=-4, energy_max=4, mesh_interval=0.1)
        >>> db = cdb.doppler_broadening(centralize_peak=False)   # projects the cached map
        >>> res = cdb.resolution()           # projects the cached map
        """
        expected_columns = ['energy_1', 'energy_2']
        if list(gamma_energy_pair.columns) != expected_columns:
            raise ValueError(
                f"Input DataFrame must contain exactly these columns in order: {expected_columns}. "
                f"Got: {gamma_energy_pair.columns.tolist()}")
        if energy_min >= energy_max:
            raise ValueError(f"energy_min ({energy_min}) must be less than energy_max ({energy_max})")

        self.pair_df = gamma_energy_pair
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.mesh_interval = mesh_interval
        self._coincidence_map = self._compute_coincidence_map()

    def _compute_coincidence_map(self):
        """
        Compute the 2D coincidence histogram (CDB) from all energy pair measurements.
        The x-axis corresponds to Doppler broadening: (E₁ − E₂)/2
        The y-axis corresponds to resolution: (E₁ + E₂ − 2·511 keV)/2

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
        res = ((cdb_pairs['energy_1'] + cdb_pairs['energy_2']) - 2 * ELECTRON_REST_MASS_KEV)/2

        # Define bin edges
        bin_edges_x = np.arange(self.energy_min, self.energy_max, self.mesh_interval)
        bin_edges_y = np.arange(self.energy_min, self.energy_max, self.mesh_interval)

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

    @property
    def coincidence_map(self) -> xr.DataArray:
        """The precomputed 2D coincidence histogram."""
        return self._coincidence_map

    def doppler_broadening(self,
                           centralize_peak: bool=True,
                           center_value: float=0):
        """
        Compute the 1D Doppler broadening spectrum by projecting the coincidence map.

        The 2D coincidence histogram is summed over the resolution axis, collapsing
        it to a function of Doppler shift only: (E₁ − E₂)/2. The result is wrapped
        in a ``PASdb`` domain for S/W parameter analysis.

        Note: unlike ``PASdb.from_spectrum``, the default ``center_value`` here is 0
        rather than 511 keV, because the CDB Doppler axis is naturally centered
        around zero by construction.

        Parameters
        ----------
        centralize_peak : bool
            If True, shifts the axis calibration so the peak center aligns with
            ``center_value``. Default is True.
        center_value : float
            The axis value the peak center should be mapped to after centralization,
            in keV. Defaults to 0.0 — the natural center of the CDB Doppler axis.
            Change this if your axis convention differs.

        Returns
        -------
        PASdb
            A PASdb domain containing the 1D Doppler broadening spectrum, ready
            for S/W parameter calculation. Poisson errors (sqrt of counts) are
            assigned automatically.

        See Also
        --------
        resolution : Project along the Doppler axis to get the resolution spectrum.
        coincidence_map : Compute the underlying 2D histogram directly.
        """
        doppler_broadening = self.coincidence_map.sum("resolution")
        doppler_broadening_spectrum = Spectrum(
            counts=doppler_broadening.values,
            counts_err=np.sqrt(doppler_broadening.values),
            axis_calib=AxisCalibration.from_array(doppler_broadening.coords["doppler"].values))

        return PASdb.from_domain(
            doppler_broadening_spectrum.domain(start_val=doppler_broadening_spectrum.axis[0],
                                               stop_val=doppler_broadening_spectrum.axis[-1]),
            centralize_peak=centralize_peak,
            center_value=center_value)

    def resolution(self):
        """
        Compute the resolution spectrum (1D) from the 2D histogram.

        This is done by summing over the Doppler axis, collapsing the 2D histogram
        to a function of total energy shift only.

        Returns
        -------
        Domain
            A Domain containing the 1D resolution spectrum (sum-energy axis).
        """
        resolution = self.coincidence_map.sum("doppler")
        resolution_spectrum = Spectrum(counts=resolution.values,
                                       counts_err=np.sqrt(resolution.values),
                                       axis_calib=AxisCalibration.from_array(resolution.coords["resolution"].values))

        return resolution_spectrum.domain(start_val=resolution_spectrum.axis[0], stop_val=resolution_spectrum.axis[-1])