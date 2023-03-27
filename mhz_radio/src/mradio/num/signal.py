"""

"""

from logging import getLogger

import numpy as np
from scipy.signal import hilbert, butter, lfilter
import scipy.fft as sf
from scipy import interpolate

logger = getLogger(__name__)


def filter_butter_band(t_series, fr_min, fr_max, f_sample):
    """
    band filter with butterfly window
    
    :param fr_min (float): [Hz] The minimal frequency of the bandpass filter
    :param fr_max (float): [Hz] The maximal frequency of the bandpass filter

    :return: filtered trace in time domain
    """
    nyq = 0.5 * f_sample 
    low = fr_min / nyq
    high = fr_max / nyq
    order = 5
    coeff_b, coeff_a = butter(order, [low, high], btype="band")
    filtered = lfilter(coeff_b, coeff_a, t_series)
    return filtered


def get_peakamptime_norm_hilbert(a2_time, a3_trace):
    """
    Get peak Hilbert amplitude norm of trace (v_max) and its time t_max without interpolation

    :param time (D,S): time, with D number of vector of trace, S number of sample
    :param traces (D,3,S): trace

    :return: t_max float(D,) v_max float(D,), norm_hilbert_amp float(D,S),
            idx_max int, norm_hilbert_amp float(D,S)
    """
    hilbert_amp = np.abs(hilbert(a3_trace, axis=-1))
    norm_hilbert_amp = np.linalg.norm(hilbert_amp, axis=1)
    # add dimension for np.take_along_axis()
    idx_max = np.argmax(norm_hilbert_amp, axis=1)[:, np.newaxis]
    t_max = np.take_along_axis(a2_time, idx_max, axis=1)
    v_max = np.take_along_axis(norm_hilbert_amp, idx_max, axis=1)
    # remove dimension (np.squeeze) to have ~vector ie shape is (n,) instead (n,1)
    return np.squeeze(t_max), np.squeeze(v_max), idx_max, norm_hilbert_amp


def get_fastest_size_fft(sig_size, f_samp_mhz, padding_fact=1):
    """

    :param sig_size:
    :param f_samp_mhz:
    :param padding_fact:

    :return: size_fft (int,0), array freq (float,1) in MHz for rfft()
    """
    assert padding_fact >= 1
    dt_s = 1e-6 / f_samp_mhz
    fastest_size_fft = sf.next_fast_len(int(padding_fact * sig_size + 0.5))
    freqs_mhz = sf.rfftfreq(fastest_size_fft, dt_s) * 1e-6
    return fastest_size_fft, freqs_mhz


def interpol_at_new_x(a_x, a_y, new_x):
    """
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    """
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)
