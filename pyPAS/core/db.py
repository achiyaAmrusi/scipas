import numpy as np
import pandas as pd
import xarray as xr
from uncertainties import ufloat, nominal_value
from pyspectrum import Peak, Spectrum, FindPeaksDomain, Convolution, gaussian_2_dev
from warnings import warn

ELECTRON_REST_MASS = 511


class PASdb(Peak):
    """
      Represents a Doppler broadening spectrum centered around the 511 keV annihilation peak.

      This class extends `Peak` from `pyspectrum` and adds tools for calculating Doppler
      broadening lineshape parameters (S and W), using uncertainty propagation via the
      `uncertainties` package.

      Parameters
      ----------
      peak_xarray : xr.DataArray
          The spectrum peak data as an xarray DataArray (with coordinates like 'channel' and values as counts).
      ubackground_l : ufloat, optional
          Background estimate from the left side of the peak (default: ufloat(0, 1)).
      ubackground_r : ufloat, optional
          Background estimate from the right side of the peak (default: ufloat(0, 1)).
      centralize_peak : bool
      to centrlize the Peak coordinates to 511 [keV] in the peak center using the Peak estimator
       (This method uses the maximal channel as the center, for better calibration use Peeak methods)
      Attributes
      ----------
      peak : xr.DataArray
          The peak data (counts vs. energy or channel).
      height_left : ufloat
          Estimated background on the left side of the peak.
      height_right : ufloat
          Estimated background on the right side of the peak.
      estimated_center : float
          Estimated centroid of the annihilation peak.
      estimated_resolution : float
          Estimated resolution (FWHM) of the annihilation peak.

      Methods
      -------
      centralize_annihilation_peak()
        centralize the annihilation peak around 511 [KeV]
      sum_with_edge_correction(e1: float, e2: float) -> float
          Integrate counts in a given interval with edge correction for partial bins.

      s_parameter_calculation(energy_domain_total, energy_domain_s) -> ufloat
          Calculate the S parameter, defined as the ratio of counts in the central peak region
          to the total counts across the entire energy domain.

      w_parameter_calculation(energy_domain_total, energy_domain_w_left, energy_domain_w_right) -> ufloat
          Calculate the W parameter, defined as the ratio of counts in the peak wings
          to the total counts across the full peak range.

      from_dataframe(cls, spectrum_data_frame, energy_calibration_poly, fwhm_calibration=None) -> PASdb
          Create a PASdb object from a pandas DataFrame containing 'channel' and 'counts' columns.

      from_spectrum(cls, spectrum: Spectrum) -> PASdb
          Create a PASdb object from a `Spectrum` instance by identifying the 511 keV peak.

      from_file(cls, spectrum_file_path, energy_calibration_poly, fwhm_calibration, sep='\\t', **kwargs) -> PASdb
          Create a PASdb object by loading data from a file (if implemented).
      """

    def __init__(self, peak_xarray: xr.DataArray, ubackground_l=ufloat(0, 1), ubackground_r=ufloat(0, 1)):
        """
        Inherit constructor from Peak.
        """
        super().__init__(peak_xarray, ubackground_l, ubackground_r)

    def centralize_annihilation_peak(self):
        """
        Centralize the annihilation peak around 511 keV.

        This method shifts the x-axis (channel/energy coordinate) of the peak
        such that the peak center aligns with the positron rest mass energy (511 keV).
        """
        # Centralize the peak
        center = self.first_moment_method_center()
        # Subtract center value from coordinate
        new_coords = self._xr_peak.coords['channel'] - (nominal_value(center) - ELECTRON_REST_MASS)
        peak_xarray = self._xr_peak.copy()
        peak_xarray.coords['channel'] = new_coords
        self._xr_peak = peak_xarray

    def sum_with_edge_correction(self, e1: float, e2: float) -> float:
        """
        Sum counts over an energy/channel range [e1, e2] in a binned xarray DataArray,
        accounting for partial bin contributions at the edges.

        This function performs an integration of the counts in `da` over the interval [e1, e2]
        using the following steps:
          - Full bins within [e1, e2] are fully included.
          - The left and right edge bins are partially included based on the fraction of their
            width that falls inside the interval.

        It assumes:
          - `da` has a coordinate (e.g. "channel") that represents bin centers.
          - Bins are uniformly spaced.
          - `da` is 1D with bin centers as the coordinate.

        Parameters
        ----------
        da : xr.DataArray
            The spectrum to integrate, indexed by a coordinate such as 'channel'.
        e1 : float
            The lower bound of the energy/channel domain.
        e2 : float
            The upper bound of the energy/channel domain.

        Returns
        -------
        float
            The total integrated counts over the interval [e1, e2],
            including edge corrections for partial bins.
        """
        peak = self.subtract_background()

        if e1 > e2:
            raise ValueError("e1 must be less than e2")

        # Slice bins that are fully inside the energy domain
        # Get bin width from existing bins (assumed uniform here)
        bin_width = (peak.channel[1] - peak.channel[0]).item()
        bins = peak.sel(channel=slice(e1 - bin_width, e2))

        # Calculate fractional contributions at edges
        left_frac = (bins.channel[1].item() - e1) / bin_width
        left_val = bins[0].item()
        left_correction = left_val * left_frac

        right_frac = (e2 - bins.channel[-1].item()) / bin_width
        right_val = bins[-1].item()
        right_correction = right_val * right_frac

        counts = bins[1:-1].sum().item() + left_correction + right_correction

        return counts

    def s_parameter_calculation(self, energy_domain_total, energy_domain_s):
        """
        Calculate the S parameter for the 511 kev peak of Spectrum according to domain definitions.

        Parameters
        ----------
        energy_domain_total: iterable  (tuple/list)
         Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        energy_domain_s: iterable  (tuple/list)
         Tuple containing the specific energy domain for S parameter calculation.

        Returns
        -------
        ufloat
         The calculated s parameter with associated uncertainty.
        """
        bin_width = (self['channel'][1] - self['channel'][0]).item()
        # The S line part
        a = self.sum_with_edge_correction(e1=energy_domain_s[0], e2=energy_domain_s[1])
        # The rest of the peak
        b = self.sum_with_edge_correction(e1=energy_domain_total[0], e2=energy_domain_s[0])
        c = self.sum_with_edge_correction(e1=energy_domain_s[1], e2=energy_domain_total[1])
        return a / (a + b + c)

    def w_parameter_calculation(self, energy_domain_total, energy_domain_w_left, energy_domain_w_right):
        """
        Calculate the W parameter for the 511 kev peak of Spectrum according to domain definitions.

        Parameters
        ----------
        energy_domain_total: iterable (tuple/list)
         Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        energy_domain_w_left: iterable (tuple/list)
        Tuple (2 index) containing the specific energy domain
         for W parameter calculation in the right wing.
        energy_domain_w_right: iterable (tuple/list)
         Tuple (2 index) containing the specific energy domain
         for W parameter calculation in the left wing.

        Returns
        -------
        ufloat
         The calculated s parameter with associated uncertainty.
        """

        bin_width = (self['channel'][1] - self['channel'][0]).item()

        e_1_l = energy_domain_w_left[0]
        e_2_l = energy_domain_w_left[1]

        e_1_r = energy_domain_w_right[0]
        e_2_r = energy_domain_w_right[1]

        e_1_peak = energy_domain_total[0]
        e_2_peak = energy_domain_total[1]

        # check input values
        if (e_1_l<e_1_peak) or (e_1_r<e_2_l) or (e_2_peak<e_2_r):
            raise  ValueError('There has been a problem with the integration boundaries')

        # The W line part
        w1 = self.sum_with_edge_correction(e1=e_1_l, e2=e_2_l)
        w2 = self.sum_with_edge_correction(e1=e_1_r, e2=e_2_r)

        # The rest of the peak (Also check that the boundaries are not too close and if so make them the same)

        # left
        if np.abs(e_1_peak-e_1_l) > bin_width:
            d1 = self.sum_with_edge_correction(e1=e_1_peak, e2=e_1_l)
        elif np.abs(e_1_peak-e_1_l) == 0:
            d1 = 0
        else:
            raise ValueError("The boundaries of W and the peak are too close.\n"
                             "To minimize computational errors, consider aligning the boundaries or increasing their separation.")
        # middle
        if np.abs(e_2_l-e_1_r) > bin_width:
            d2 = self.sum_with_edge_correction(e1=e_2_l, e2=e_1_r)
        elif np.abs(e_2_l - e_1_r) == 0:
            d2 = 0
        else:
            raise ValueError("The boundaries of W and the peak are too close.\n"
                             "To minimize computational errors, consider aligning the boundaries or increasing their separation.")
        # right
        if np.abs(e_1_r-e_2_peak) > bin_width:
            d3 = self.sum_with_edge_correction(e1=e_1_r, e2=e_2_peak)
        elif np.abs(e_1_r - e_2_peak) == 0:
            d3 = 0
        else:
            raise ValueError("The boundaries of W and the peak are too close.\n"
                             "To minimize computational errors, consider aligning the boundaries or increasing their separation.")

        return (w1+w2)/(w1+w2+d1+d2+d3)


    @classmethod
    def from_dataframe(cls, spectrum_data_frame: pd.DataFrame,
                       energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None):
        """
        load spectrum from a dataframe which has 2 columns
        first column is the channel and the second is counts
        function return Spectrum

        Parameters
        ----------
        spectrum_data_frame: pd.DataFrame
         spectrum in form of a dataframe such that the column are -  'channel', 'counts'
        energy_calibration_poly: numpy.poly1d([a, b])
        the energy calibration of the detector
        fwhm_calibration: Callable
        a function that given energy/channel(first raw in file) returns the fwhm

        Returns
        -------
        PASdb
        core spectrum from the file in PASdb class .
        """
        # Load the pyspectrum file in form of DataFrame
        spectrum = Spectrum.from_dataframe(spectrum_data_frame, energy_calibration_poly=energy_calibration_poly,
                                           fwhm_calibration=fwhm_calibration)
        estimated_fwhm_ch = lambda ch: fwhm_calibration(energy_calibration_poly(ch)) / energy_calibration_poly[1]
        convolution = Convolution(estimated_fwhm_ch, gaussian_2_dev)
        find_peaks = FindPeaksDomain(spectrum, convolution)
        peak = find_peaks.to_peak(ELECTRON_REST_MASS)
        return PASdb(peak._xr_peak, peak.height_left, peak.height_right)

    @classmethod
    def from_spectrum(cls, spectrum: Spectrum):
        """
        load spectrum, look for the 511 peak and return it

        Parameters
        ----------
        spectrum: pd.DataFrame
         spectrum object with annhilation peak

        Returns
        -------
        PASdb
        core spectrum from the file in PASdb class .
        """
        # Load the pyspectrum file in form of DataFrame

        convolution = Convolution(spectrum.fwhm_calibration, gaussian_2_dev)
        find_peaks = FindPeaksDomain(spectrum, convolution, n_sigma_threshold=5)
        peak = find_peaks.to_peak(ELECTRON_REST_MASS)
        return PASdb(peak._xr_peak, peak.height_left, peak.height_right)
