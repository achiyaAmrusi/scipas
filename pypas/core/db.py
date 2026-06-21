import warnings
import numpy as np
import xarray as xr
from uncertainties import nominal_value
from scispectrum.core import Domain, Spectrum
from scispectrum.calibration import AxisCalibration
from scispectrum.identification import SNRFinder, Convolution, gaussian_2_dev
from scispectrum.domain_analysis.single_peak import center_estimator, sum_under
from pypas.core.const import ELECTRON_REST_MASS_KEV

class PASdb(Domain):
    """
      Represents a Doppler broadening spectrum centered around the 511 keV annihilation peak.

      This class extends `Domain` from `scispectrum` and adds tools for calculating Doppler
      broadening lineshape parameters (S and W), using uncertainty propagation via the
      `uncertainties` package.

      Parameters
      ----------
      spectrum: Spectrum
        The full spectrum with the annihilation peak
      start: int
        The start index of the annihilation peak
      stop: int
       The stop index of the annihilation peak
      centralize_peak : bool
      to centrlize the Peak coordinates to 511 [keV] in the peak center using the center estimator
      from scispectrum.domain_analysis


      Attributes
      ----------
      data : xr.DataArray
          Domain counts as an xarray DataArray, with background
          subtracted if provided.
      data_with_errors : xr.DataArray
          Domain counts with uncertainty via the uncertainties library.
      background : np.ndarray or None
          The background array, if set.
      indices : np.ndarray
          Array of spectrum indices covered by this domain.

      Methods
      -------
      recenter(center_value) -> None
          Shift the axis calibration in-place so the annihilation peak center aligns
          with ``center_value`` (default: electron rest mass energy, ~511 keV).

      s_parameter_calculation(energy_domain_total, energy_domain_s) -> ufloat
          Calculate the S parameter, defined as the ratio of counts in the central peak region
          to the total counts across the entire energy domain.

      w_parameter_calculation(energy_domain_total, energy_domain_w_left, energy_domain_w_right) -> ufloat
          Calculate the W parameter, defined as the ratio of counts in the peak wings
          to the total counts across the full peak range.

      from_domain(cls, domain, centralize_peak, center_value) -> PASdb
          Wrap an existing ``Domain`` as a ``PASdb``, optionally recentering the axis.

      from_spectrum(cls, spectrum, ...) -> PASdb
          Locate the 511 keV annihilation peak in a ``Spectrum`` automatically and
          return it as a ``PASdb`` domain.

      """

    def __init__(self, spectrum: Spectrum,
                 start: int,
                 stop: int,
                 background: np.ndarray=None,
                 centralize_peak: bool=True,
                 center_value: float=ELECTRON_REST_MASS_KEV):
        """
        Inherit constructor from Domain
         If centralize_peak centrlize the calibration is adjusted so peak center is on 511 KeV.
        Parameters
        ----------
        spectrum : Spectrum
            The full spectrum containing the annihilation peak.
        start : int
            Start index of the annihilation peak domain.
        stop : int
            Stop index of the annihilation peak domain.
        background : np.ndarray, optional
            Background array to subtract.
        centralize_peak : bool
            If True, shifts the axis calibration so the peak center aligns
            with electron_rest_mass_value.
        center_value  : float
            The axis value the peak center should be mapped to after centralization.
            Defaults to the electron rest mass energy in keV (510.99895 keV).
            For example, user can use 0.0 for CDB or momentum spectra which are naturally centered around zero.

        Examples
        --------
        >>> import numpy as np
        >>> from scispectrum import Spectrum, Domain, AxisCalibration, ResolutionCalibration
        >>> bins = np.linspace(511-100, 511+100, 1000)
        >>> centers = (bins[1:] + bins[:-1])/2
        >>> list_counts = np.random.normal(loc=511, scale=5, size=1000000)
        >>> counts, _ = np.histogram(list_counts, bins=bins)
        >>> counts_with_noise = np.random.poisson(counts + 1000)
        >>> resolution = ResolutionCalibration(lambda e: 5*2*np.sqrt(2*np.log(2)))
        >>> spec = Spectrum(counts=counts_with_noise,
        ...          counts_err=np.sqrt(counts_with_noise),
        ...          axis_calib=AxisCalibration.from_array(centers),
        ...          resolution_calib=resolution)
        >>> db = PASdb.from_spectrum(spec)
        """
        super().__init__(spectrum=spectrum, start=start, stop=stop, background=background)
        if centralize_peak:
            self.recenter(center_value)

    def recenter(self, center_value: float=ELECTRON_REST_MASS_KEV):
        """
        Shift the axis calibration so the annihilation peak center aligns with
        the electron rest mass energy.

        The peak center is estimated using the weighted centroid method from
        ``scispectrum.domain_analysis.single_peak.center_estimator``. The axis
        calibration of the parent Spectrum is then shifted so that the estimated
        center maps to ``electron_rest_mass_value``.

        Note: this modifies the axis calibration of the parent Spectrum in-place,
        which affects all domains derived from it.

        Parameters
        ----------
        center_value  : float
            The axis value the peak center should be mapped to after centralization.
            Defaults to the electron rest mass energy in keV (510.99895 keV).
            For example, user can use 0.0 for CDB or momentum spectra which are naturally centered around zero.

        Warns
        -----
        UserWarning
            If the peak center estimation is unreliable due to a noisy or
            asymmetric peak — propagated from ``center_estimator``.
        """
        # calculates the peak center
        center = nominal_value(center_estimator(self))
        # recalibrate
        shift = center - center_value
        old_calib = self.spectrum.axis_calib  # capture current calibration
        new_axis_calibration = AxisCalibration(
            lambda e: old_calib.apply(e) - shift,  # closes over old_calib, not self.spectrum.axis_calib
            name=self.spectrum.axis_name)
        self.spectrum.set_axis_calibration(new_axis_calibration)


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
        # The S line part
        a = sum_under(single_peak_domain=self, left_edge=energy_domain_s[0], right_edge=energy_domain_s[1])
        # The rest of the peak
        b = sum_under(single_peak_domain=self, left_edge=energy_domain_total[0], right_edge=energy_domain_s[0])
        c = sum_under(single_peak_domain=self, left_edge=energy_domain_s[1], right_edge=energy_domain_total[1])
        return a / (a + b + c)

    def w_parameter_calculation(self, energy_domain_total, energy_domain_w_left, energy_domain_w_right):
        """
        Calculate the W parameter for the 511 kev peak of Spectrum according to domain definitions.

        Parameters
        ----------
        energy_domain_total: iterable (tuple/list)
         Tuple containing the total energy domain of interest of the defect parameter
         calculation (e.g., (E1, E2)).
        energy_domain_w_left : iterable (tuple/list)
            (e_low, e_high) integration window for the left wing of the W parameter.
        energy_domain_w_right : iterable (tuple/list)
            (e_low, e_high) integration window for the right wing of the W parameter.

        Returns
        -------
        ufloat
         The calculated w parameter with associated uncertainty.
        """

        bin_width = (self.spectrum.axis[1]-self.spectrum.axis[0])

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
        w1 = sum_under(single_peak_domain=self, left_edge=e_1_l, right_edge=e_2_l)
        w2 = sum_under(single_peak_domain=self, left_edge=e_1_r, right_edge=e_2_r)

        # The rest of the peak — gaps smaller than one bin are sub-resolution.
        # A sub-bin gap is small but not zero in PAS; warn and still compute.

        # left: region between peak edge and left W window
        d1_gap = np.abs(e_1_peak - e_1_l)
        if d1_gap == 0:
            d1 = 0
        else:
            if d1_gap < bin_width:
                warnings.warn(
                    f"Left gap between peak boundary and W window ({d1_gap:.4f} keV) "
                    f"is smaller than one bin ({bin_width:.4f} keV). "
                    "Sub-bin integration may have limited accuracy.",
                    UserWarning, stacklevel=2)
            d1 = sum_under(single_peak_domain=self, left_edge=e_1_peak, right_edge=e_1_l)

        # middle: S region between the two W windows
        d2_gap = np.abs(e_2_l - e_1_r)
        if d2_gap == 0:
            d2 = 0
        else:
            if d2_gap < bin_width:
                warnings.warn(
                    f"Middle gap between W windows ({d2_gap:.4f} keV) "
                    f"is smaller than one bin ({bin_width:.4f} keV). "
                    "Sub-bin integration may have limited accuracy.",
                    UserWarning, stacklevel=2)
            d2 = sum_under(single_peak_domain=self, left_edge=e_2_l, right_edge=e_1_r)

        # right: region between right W window and peak edge
        d3_gap = np.abs(e_2_r - e_2_peak)
        if d3_gap == 0:
            d3 = 0
        else:
            if d3_gap < bin_width:
                warnings.warn(
                    f"Right gap between W window and peak boundary ({d3_gap:.4f} keV) "
                    f"is smaller than one bin ({bin_width:.4f} keV). "
                    "Sub-bin integration may have limited accuracy.",
                    UserWarning, stacklevel=2)
            d3 = sum_under(single_peak_domain=self, left_edge=e_2_r, right_edge=e_2_peak)

        return (w1+w2)/(w1+w2+d1+d2+d3)

    @classmethod
    def from_domain(cls, domain: Domain,
                    centralize_peak: bool = True,
                    center_value : float = ELECTRON_REST_MASS_KEV):
        """
        Wrap an existing ``Domain`` as a ``PASdb``, optionally recentering the axis.

        Parameters
        ----------
        domain : Domain
            A domain already sliced around the annihilation peak.
        centralize_peak : bool
            If True, shifts the axis calibration so the peak center aligns
            with electron_rest_mass_value.
        center_value  : float
            The axis value the peak center should be mapped to after centralization.
            Defaults to the electron rest mass energy in keV (510.99895 keV).
            For example, user can use 0.0 for CDB or momentum spectra which are naturally centered around zero.

        Returns
        -------
        PASdb
            The domain recast as a ``PASdb`` instance, with axis optionally recentered.
        """
        return PASdb(domain.spectrum,
                     domain.start,
                     domain.stop,
                     domain.background,
                     centralize_peak=centralize_peak,
                     center_value=center_value)

    @classmethod
    def from_spectrum(cls, spectrum: Spectrum,
                      window_fwhm: float = 3.,
                      n_sigma_signal_threshold=5,
                      n_sigma_bg_threshold=2.0,
                      persistence_factor=0.5,
                      centralize_peak: bool = True):
        """
        Identify the 511 keV annihilation peak in a spectrum and return it as a PASdb domain.

        The peak is located automatically using the SNR-based peak finder
        (``scispectrum.identification.SNRFinder``) with a Gaussian second derivative
        convolution kernel. The spectrum must have a resolution calibration set,
        as it is used to scale the convolution window.

        If ``centralize_peak`` is True, the axis calibration of the parent spectrum
        is shifted in-place so the peak center aligns with the electron rest mass energy.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum object containing the 511 keV annihilation peak.
            Must have a resolution calibration set.
        window_fwhm : float
            Width of the convolution window in units of the local FWHM.
            Larger values smooth more aggressively. Default is 3.
        n_sigma_signal_threshold : float
            Number of standard deviations above background required to
            identify a peak. Higher values reduce false positives. Default is 5.
        n_sigma_bg_threshold : float
            Number of standard deviations used to define the background level.
            Default is 2.0.
        persistence_factor : float
            Controls how persistent a peak must be across scales to be accepted.
            Default is 0.5.
        centralize_peak : bool
            If True, shifts the spectrum axis calibration in-place so the
            detected peak center aligns with the electron rest mass energy.
            Default is True.

        Returns
        -------
        PASdb
            A PASdb domain centered on the identified 511 keV annihilation peak.

        Raises
        ------
        ValueError
            If no peak is found near the electron rest mass energy in the spectrum.
        """
        # Load the scispectrum file in form of DataFrame

        convolution = Convolution(spectrum.resolution_calib.apply,
                                  gaussian_2_dev,
                                  window_fwhm=window_fwhm)
        find_peaks = SNRFinder(convolution=convolution,
                               n_sigma_signal_threshold=n_sigma_signal_threshold,
                               n_sigma_bg_threshold=n_sigma_bg_threshold,
                               persistence_factor=persistence_factor)
        domain = find_peaks.domain(spectrum, ELECTRON_REST_MASS_KEV)
        return PASdb.from_domain(domain, centralize_peak=centralize_peak)
