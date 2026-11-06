import numpy as np
import pandas as pd
import pytest

from scipas.filter import PasCoincidenceFilter


def test_single_pair_at_boundary():
    """Last (and only) detector-2 event must be compared, not silently dropped."""
    det1 = pd.DataFrame({'time': [100], 'channel': [512]})
    det2 = pd.DataFrame({'time': [100], 'channel': [508]})

    result = PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)

    assert len(result) == 1
    assert result['channel_1'].iloc[0] == 512
    assert result['channel_2'].iloc[0] == 508


def test_multiple_matches_single_det1_event():
    """One detector-1 event matching several detector-2 events must not overflow the buffer."""
    det1 = pd.DataFrame({'time': [100], 'channel': [1]})
    det2 = pd.DataFrame({'time': [101, 102, 103], 'channel': [21, 22, 23]})

    result = PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)

    assert len(result) == 3
    assert list(result['channel_1']) == [1, 1, 1]
    assert list(result['channel_2']) == [21, 22, 23]


def test_last_elements_both_coincident():
    """Both inputs' last rows being coincident must not raise, and must be included."""
    det1 = pd.DataFrame({'time': [100, 500], 'channel': [1, 2]})
    det2 = pd.DataFrame({'time': [100, 505], 'channel': [11, 12]})

    result = PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)

    assert len(result) == 2
    assert list(result['channel_1']) == [1, 2]
    assert list(result['channel_2']) == [11, 12]


def test_energy_column_input():
    det1 = pd.DataFrame({'time': [100], 'energy': [511.0]})
    det2 = pd.DataFrame({'time': [100], 'energy': [511.0]})

    result = PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)

    assert list(result.columns) == ['energy_1', 'energy_2']
    assert result['energy_1'].iloc[0] == 511.0
    assert result['energy_2'].iloc[0] == 511.0


def test_mixed_matches_baseline():
    """Docstring-style example: some matches, some misses, no trailing overflow or drop."""
    det1 = pd.DataFrame({
        'time': [100, 200, 300, 400, 500],
        'channel': [512, 498, 523, 480, 510]})
    det2 = pd.DataFrame({
        'time': [102, 205, 350, 401, 600],
        'channel': [508, 501, 490, 519, 475]})

    result = PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)

    assert list(result['channel_1']) == [512, 498, 480]
    assert list(result['channel_2']) == [508, 501, 519]
    assert result['channel_1'].dtype == np.float64
    assert result['channel_2'].dtype == np.float64


def test_no_coincidence():
    det1 = pd.DataFrame({'time': [100], 'channel': [1]})
    det2 = pd.DataFrame({'time': [1000], 'channel': [2]})

    result = PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)

    assert len(result) == 0
    assert result['channel_1'].dtype == np.float64
    assert result['channel_2'].dtype == np.float64


def test_column_name_mismatch_raises():
    det1 = pd.DataFrame({'time': [100], 'channel': [1]})
    det2 = pd.DataFrame({'time': [100], 'energy': [511.0]})

    with pytest.raises(ValueError):
        PasCoincidenceFilter.time_coincidence_filter(det1, det2, max_time_interval=10)
