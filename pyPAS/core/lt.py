from pyspectrum import Spectrum
from abc import ABC, abstractmethod
import numpy as np

class TimeResolution(ABC):
    """
    Abstract base class for time-domain resolution (instrument response) modeling.

    Conceptual Meaning
    ------------------
    TimeResolution represents the detector or measurement system response function
    that distorts the ideal physical time spectrum before observation.

    In positron fit spectroscopy, the measured spectrum is typically modeled as:

        Measured_signal(t) = Convolution[ True_physics_signal(t), resolution(t') ]

    where resolution(t') is the instrument response function.

    This class defines the interface for evaluating the resolution function
    and performing numerical convolution with a physical signal.
    Based on this interface, various resolution types can be defined.
    pyPAS currently provides two types of resolution clsses -
    - MeasuredRF, measured based resolution function
    - MultiGaussianRF, gausian fit based resolution

    Methods
    -------
    evaluate(time: np.ndarray) -> np.ndarray
        Returns the instrument response function evaluated on the time axis.

        Parameters
        ----------
        time : np.ndarray
            Time grid on which the resolution function is evaluated.

        Returns
        -------
        np.ndarray
            Resolution function values on the same grid.

    convolve(signal: np.ndarray, time: np.ndarray) -> np.ndarray
        Performs numpy convolution between physical signal and RF.

        Parameters
        ----------
        signal : np.ndarray
            Ideal physics signal before detector distortion.

        time : np.ndarray
            Time axis corresponding to signal discretization.

        Returns
        -------
        np.ndarray
            Convolved signal truncated to original signal length.

    Notes
    -----
    - The convolution is performed in the time domain.
    - The implementation assumes constant grid spacing.
    """
    @abstractmethod
    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Returns resolution function evaluated on time axis.
        """
        pass

    def convolve(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Numerically convolve physical signal with instrument response function.

        This provides the default forward-model distortion operation.

        Parameters
        ----------
        signal : np.ndarray
            Ideal physical signal before detector response distortion.

        t : np.ndarray
            Time grid corresponding to signal discretization.

        Returns
        -------
        np.ndarray
            Convolved signal.

        Notes
        -----
        - Assumes uniform time spacing.
        - Output is truncated to match input signal length.
        """
        irf = self.evaluate(t)
        dt = t[1] - t[0]
        zero_point_index = np.where(t>0)[0][0]

        return np.convolve(signal, irf, mode="full")[zero_point_index:len(t)+zero_point_index] * dt


class MeasuredRF(TimeResolution):
    """
    Resolution function constructed directly from measured spectrum data.
    This class wraps experimentally measured detector response spectra.

    Parameters
    ----------
    spectrum : Spectrum
        Spectroscopy spectrum container holding detector response counts.

    Methods
    -------
    evaluate(t: np.ndarray) -> np.ndarray
        Returns normalized detector response counts.

    convolve(signal: np.ndarray, time: np.ndarray) -> np.ndarray
        Performs numpy convolution between physical signal and RF.

    Notes
    -----
    - The spectrum counts are normalized to unit integral.
    """
    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum
        self.spectrum.counts = self.spectrum.counts/self.spectrum.counts.sum()

    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Evaluate multi-Gaussian resolution function.

        Parameters
        ----------
        time : np.ndarray
         Time axis.

        Returns
        -------
        np.ndarray
            Normalized resolution function.
        """
        return self.spectrum.counts


class MultiGaussianRF(TimeResolution):
    """
    Multi-component Gaussian Instrument Response Function.

    Mathematical Model
    ------------------
    The resolution function is modeled as a weighted mixture of Gaussian kernels:

        IRF(t) = Σ_i w_i exp(-(t - t0_i)^2 / (2 σ_i^2))

    Parameters
    ----------
    sigmas : np.ndarray
        Standard deviations of Gaussian components.
    weights : np.ndarray
        Mixing weights of Gaussian components.
        The weights are normalized in the initialization
    t0 : np.ndarray
        Center offsets of Gaussian components.

    """

    def __init__(self, sigmas: np.ndarray, weights: np.ndarray, t0: np.ndarray):
        self.sigmas = sigmas
        self.weights = weights
        self.t0 = t0
        if (self.sigmas.ndim != 1 or len(self.sigmas) == 0) or (self.weights.ndim != 1 or len(self.weights) == 0)  or (self.t0.ndim != 1 or len(self.t0) == 0):
            raise ValueError("sigmas and weights must be nonempty 1D")

        if not (len(self.sigmas) == len(self.weights) == len(self.t0)):
            raise ValueError("sigmas, t0 and weights must have same length")

        if np.any(self.sigmas <= 0) or np.any(self.weights < 0):
            raise ValueError("All sigmas must be positive")

        self.weights = self.weights / self.weights.sum()  # normalize


    def evaluate(self, time: np.ndarray) -> np.ndarray:
        """
        Evaluate multi-Gaussian resolution function.

        Parameters
        ----------
        time : np.ndarray
         Time axis.

        Returns
        -------
        np.ndarray
            Normalized resolution function.
        """
        irf = np.zeros_like(time, dtype=float)
        sigma = np.vstack(self.sigmas)
        weight = np.vstack(self.weights)
        t_center = np.vstack(self.t0)
        components = weight * np.sqrt(1/(2*np.pi*sigma**2)) * np.exp(-(time - t_center)**2 / (2 * sigma**2))
        # Normalize numerically (important for discrete grid)
        return components.sum(axis=0)


class PASLifetime:
    """"
    Represents a fit spectrum.
    This class is the basis for the fit analysis.
    """

    def __init__(self, lifetime: Spectrum, resolution: TimeResolution):
        self.lifetime = lifetime
        self.resolution = resolution