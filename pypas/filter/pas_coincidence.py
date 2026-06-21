import numpy as np
import pandas as pd
from scispectrum import AxisCalibration
from pypas.core.const import ELECTRON_REST_MASS_KEV

class PasCoincidenceFilter:
    """
    Coincidence filtering utility for Positron Annihilation Spectroscopy (PAS).

    This class provides methods to process data from two detectors in PAS experiments,
    identifying pairs of events that are both close in time and consistent with
    positron annihilation energy expectations.

    Supported input formats include either:
    - Time and channel data (with optional energy calibration), or
    - Time and calibrated energy data.

    Energy filtering is based on the sum of energies being close to 2 * ELECTRON_REST_MASS,
    within a Gaussian window defined by detector resolutions (FWHM).

    Notes
    -----
    - Energy filtering is done in sigma units:
        σ = FWHM / (2*sqrt(2 * ln(2)))

    - Either 'channel' or 'energy' column must be present alongside 'time' in input DataFrames.

    Methods
    -------
    time_coincidence_filter(cls, time_channel_1: pd.DataFrame, time_channel_2: pd.DataFrame, max_time_interval=10)
        Identify pairs of events from two detectors that occurred within a time window.

    energy_coincidence_filter(cls, coincidence_events: pd.DataFrame, ...)
        Identify energy-coincident pairs based on the energy sum being close to 1022 keV (2 * 511 keV),
        using calibrated energy columns or applying calibration on-the-fly from channel columns.

    _get_value_column(cls, df: pd.DataFrame, label: str)
        Internal utility to infer whether the data uses 'channel' or 'energy' as the value column.
    """
    def __init__(self):
        pass

    @classmethod
    def time_coincidence_filter(cls, time_channel_1: pd.DataFrame, time_channel_2: pd.DataFrame, max_time_interval=10):
        """
        Filters two DataFrames to find time-coincident events.

        For each timestamp in detector 1, the function checks whether there is a corresponding
        timestamp in detector 2 that is within the `max_time_interval`.
        If such a pair exists, the channel/energy values are saved.

        Parameters
        ----------
        time_channel_1 : pd.DataFrame
            A DataFrame with columns ['time', 'channel'] or ['time', 'energy'].
        time_channel_2 : pd.DataFrame
            Same format as above, for the second detector.
        max_time_interval : float
            Maximum allowed time difference for coincidence (in the same units as 'time').

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['channel_1', 'channel_2'] representing the coincident pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scispectrum import AxisCalibration
    >>> from pypas.filter import PasCoincidenceFilter
    >>> det_1 = pd.DataFrame({
    ... 'time':    [100, 200, 300, 400, 500],
    ... 'channel': [512, 498, 523, 480, 510]})
    >>> det_2 = pd.DataFrame({
    ... 'time':    [102, 205, 350, 401, 600],
    ... 'channel': [508, 501, 490, 519, 475]})
    >>> time_window = 10  # time units
    >>> coincident_pairs = PasCoincidenceFilter.time_coincidence_filter(det_1, det_2, max_time_interval=time_window)
        """
        # Get the second column names from both inputs
        col1 = cls._get_value_column(time_channel_1, "time_channel_1")
        col2 = cls._get_value_column(time_channel_2, "time_channel_2")
        if col1 != col2:
            raise ValueError(f"Second column name mismatch: '{col1}' vs '{col2}'")

        # 2D vector for the coincidence events (The array has the maximal size possible for the data)
        time_coincidence_events = np.zeros((time_channel_1.shape[0],2))

        # initialization of the index parameters
        coincidence_index = 0
        index_2 = 0
        index_1 = 0
        index_2_lim = time_channel_2.shape[0] - 1

        time_2 = time_channel_2['time'].iloc[index_2]

        for index_1, time_1 in enumerate(time_channel_1['time']):
            # check coincidence
            is_coin = abs(time_1 - time_2) < max_time_interval
            while index_2 < index_2_lim and (time_1 >= time_2 or is_coin):
                # if coincidence, save events and go on
                if is_coin:
                    time_coincidence_events[coincidence_index]  = np.array([time_channel_1[col1][index_1],
                                                                            time_channel_2[col2][index_2]])
                    coincidence_index = coincidence_index + 1
                index_2 = index_2 + 1
                time_2 = time_channel_2['time'][index_2]
                # check coincidence of the next
                is_coin = abs(time_1 - time_2) < max_time_interval

        return pd.DataFrame({col1+'_1': time_coincidence_events[:coincidence_index, 0],
                             col1+'_2':  time_coincidence_events[:coincidence_index, 1]})

    @classmethod
    def energy_coincidence_filter(cls, coincidence_events: pd.DataFrame,
                                  axis_calibration_1: AxisCalibration, axis_calibration_2: AxisCalibration,
                                  local_fwhm_1=1, local_fwhm_2=1, number_of_cdb_sigma=3):
        """
        Going through the time and energy stamps of 2 detectors data looking for coincidence pair.
        if the counts pair are valid coincidence measurement, the function saves it.

        Parameters
        ----------
        det_1_time_channel, det_2_time_channel: pd.dataframe
        a table of time - channel, also time - channel - alert-flag is optional
        axis_calibration_1, axis_calibration_2: AxisCalibration
        the energy calibration of the detector
        local_fwhm_1: float
         energy resolution of detector 1 in the annihilation peak (FWHM)
        local_fwhm_2: float
        energy resolution of detector 1 in the annihilation peak (FWHM)

        Returns
        -------
        pd.DataFrame
        contain the coincidence pairs where the dataframe columns are ['energy_1, energy_2]
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scispectrum import AxisCalibration
    >>> from pypas.filter import PasCoincidenceFilter
    >>> det_1 = pd.DataFrame({
    ... 'time':    [100, 200, 300, 400, 500],
    ... 'channel': [512, 498, 523, 480, 510]})
    >>> det_2 = pd.DataFrame({
    ... 'time':    [102, 205, 350, 401, 600],
    ... 'channel': [508, 501, 490, 519, 475]})
    >>> time_window = 10  # time units
    >>> coincident_pairs = PasCoincidenceFilter.time_coincidence_filter(det_1, det_2, max_time_interval=time_window)
    >>> energy_pairs = PasCoincidenceFilter.energy_coincidence_filter(
    ... coincidence_events=coincident_pairs,
    ... axis_calibration_1=AxisCalibration(lambda ch: ch, name="energy_keV"),
    ... axis_calibration_2=AxisCalibration(lambda ch: ch, name="energy_keV"),
    ... local_fwhm_1=1.2,
    ... local_fwhm_2=1.2,
    ... number_of_cdb_sigma=3)
        """

        # taks the data in a copy
        columns = coincidence_events.columns
        # Determine whether we have raw channels or already-calibrated energy
        if 'channel_1' in columns and 'channel_2' in columns:
            calibrate_coincidence_events = pd.DataFrame({'energy_1':axis_calibration_1.apply(coincidence_events['channel_1']),
                                                         'energy_2':axis_calibration_2.apply(coincidence_events['channel_2'])})
        elif 'energy_1' in columns and 'energy_2' in columns:
            # Already calibrated — just continue
            calibrate_coincidence_events = coincidence_events.copy()
            pass
        else:
            raise ValueError(
                "Expected either ['channel_1', 'channel_2'] or ['energy_1', 'energy_2'] in input DataFrame.")

        # The resolution deviation
        coincidence_energy_test = np.abs(calibrate_coincidence_events['energy_1'] +\
                                         calibrate_coincidence_events['energy_2']- 2*ELECTRON_REST_MASS_KEV)
        # The maximal resolution deviation allowed
        sig_1 = local_fwhm_1 / (2 * np.sqrt(2 * np.log(2)))
        sig_2 = local_fwhm_2 / (2 * np.sqrt(2 * np.log(2)))
        max_dev = number_of_cdb_sigma*(sig_2 ** 2 + sig_1 ** 2) ** 0.5

        calibrate_coincidence_events = calibrate_coincidence_events[coincidence_energy_test<max_dev]
        return calibrate_coincidence_events.reset_index(drop=True)


    @classmethod
    def _get_value_column(cls, df, label):
        """
        Infers the name of the non-time value column in a time-channel or time-energy DataFrame.

        This helper function ensures the DataFrame contains a 'time' column and either a 'channel' or 'energy' column.
        It returns the name of the value column (either 'channel' or 'energy').

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame expected to contain a 'time' column and one of 'channel' or 'energy'.
        label : str
            A label for the dataset (used in error messages) to help identify the problematic input.
        Returns
        -------
        str
            The name of the value column, either 'channel' or 'energy'.
        Raises
        ------
        ValueError
            If the 'time' column is missing or if neither 'channel' nor 'energy' is found.
        """
        if 'time' not in df.columns:
            raise ValueError(f"{label} must have a 'time' column.")
        for key in ['channel', 'energy']:
            if key in df.columns:
                return key
        raise ValueError(f"{label} must contain either a 'channel' or 'energy' column.")
