"""

"""

from logging import getLogger

import numpy as np
from scipy.signal import hilbert, butter, lfilter, freqz, filtfilt
import scipy.fft as sf
from scipy import interpolate
import matplotlib.pylab as plt

logger = getLogger(__name__)


def filter_butter_band_fft(t_series, fr_min, fr_max, f_sample):
    """
    band filter with butterfly window with fft method, seems equivalent to
        filtered = filtfilt(coeff_b, coeff_a, t_series, axis=0)

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 6
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    fastest_size_fft, freqs_mhz = get_fastest_size_fft(t_series.shape[0], f_sample)
    plt.figure()
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=freqs_mhz.shape[0])
    print(w.shape)
    plt.plot(w * 1e-6, abs(h), ".")
    plt.grid()
    abs_h = abs(h)
    # abs_h /= abs_h.sum()
    # filtered = lfilter(coeff_b, coeff_a, t_series, axis=0)
    # filtered = filtfilt(coeff_b, coeff_a, t_series, axis=0)
    f_fft = sf.rfft(t_series.T, fastest_size_fft) * abs_h
    filtered = sf.irfft(f_fft)[:, : t_series.shape[0]]
    print("filtered")
    print(filtered.shape)
    return filtered.T


def filter_butter_band_lfilter(t_series, fr_min, fr_max, f_sample):
    """
    band filter with butterfly window

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 6
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    filtered = lfilter(coeff_b, coeff_a, t_series, axis=0)
    print(filtered.shape)
    return filtered


def filter_butter_band_causal(t_series, fr_min, fr_max, f_sample, f_plot=False):
    """
    passband filter **causal** with butterfly window with fft

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 6
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=t_series.shape[0])
    if f_plot:
        plt.figure()
        plt.title(f"Power sprectum of Butterworth band filter [{fr_min}, {fr_max}]")
        plt.plot(w * 1e-6, abs(h), label="no causal")
        plt.xlabel("MHz")
    # add causality condition
    #    add minus to have the signal in right direction, why ?
    h.imag = -hilbert(np.real(h)).imag
    if f_plot:
        print(w.shape)
        plt.plot(w * 1e-6, abs(h), ".", label="causal")
        plt.grid()
        plt.legend()
    f_fft = sf.fft(t_series.T) * h
    filtered = sf.ifft(f_fft)
    print("filtered:", filtered.shape)
    return np.real(filtered.T)


def filter_butter_band_causal_hc(t_series, fr_min, fr_max, f_sample, f_plot=False):
    """
    passband filter **causal** with butterfly window with fft

    :return: filtered trace in time domain
    """
    low = fr_min * 1e6
    high = fr_max * 1e6
    f_hz = f_sample * 1e6
    print(f_hz, low, high)
    order = 6
    coeff_b, coeff_a = butter(order, [low, high], btype="bandpass", fs=f_hz)
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=t_series.shape[0])
    if f_plot:
        plt.figure()
        plt.title(f"Power sprectum of Butterworth band filter [{fr_min}, {fr_max}]")
        plt.plot(w * 1e-6, abs(h), label="no causal")
        plt.xlabel("MHz")
    # add causality condition
    #    add minus to have the signal in right direction, why ?
    h.imag = -hilbert(np.real(h)).imag
    size_h = w.shape[0]
    h_hc = h[:size_h//2+1]
    if f_plot:
        print(w.shape)
        plt.plot(w * 1e-6, abs(h), ".", label="causal")
        plt.grid()
        plt.legend()
    f_fft = sf.rfft(t_series.T) * h_hc
    filtered = sf.irfft(f_fft)
    print("filtered:", filtered.shape)
    return filtered.T


def patch_leff(fft_leff_hc, f_hz):
    """
    ringing and causality
     
    :param fft_leff_hc:
    :type fft_leff_hc:
    :param f_hz:
    :type f_hz:
    """
    print(f_hz)
    coeff_b, coeff_a = butter(6, [60*1e6, 220*1e6], btype="bandpass", fs=f_hz)
    size_fc = 2*(fft_leff_hc.shape[0]-1)
    print(fft_leff_hc.shape, size_fc)
    w, h = freqz(coeff_b, coeff_a, fs=f_hz, worN=fft_leff_hc.shape[0], include_nyquist=True)
    if False:
        print("w : ",w, w.shape)
        size_h = w.shape[0]
        h_hc = h[:size_h//2+1]
        fft_leff_hc_cor = fft_leff_hc*h
    else:
        # add causality
        fc_leff = np.concatenate((fft_leff_hc, np.flip(np.conj(fft_leff_hc[1:-1]))))
        fc_leff.imag = -hilbert(np.real(fc_leff)).imag
        size_l = fc_leff.shape[0]
        fft_leff_hc_cor = fc_leff[:size_l//2+1]
    if True:
        plt.figure()
        plt.title("Leff correction")
        plt.plot(w * 1e-6, abs(fft_leff_hc), label="leff")
        plt.plot(w * 1e-6, abs(fft_leff_hc_cor), "-.", label="leff cor")
        plt.plot(w * 1e-6, abs(h), label="Butterworth")
        plt.grid()
        plt.legend()
    return fft_leff_hc_cor
    
    
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
