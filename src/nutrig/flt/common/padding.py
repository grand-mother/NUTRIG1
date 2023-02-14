'''
Created on Oct 25, 2022

@author: root
'''

import numpy as np
import scipy.signal as ss
import scipy.fft as sf
import matplotlib.pyplot as plt
from flt.common import basis


def create_pseudo_signal(size, a=-20, b=80):
    assert size > 50
    s_zone = int(size * 0.7)
    win_1 = ss.bohman(s_zone) * a
    win_2 = ss.kaiser(s_zone, 14) * b
    sig = np.zeros(size, dtype=np.float32)
    sig[10:10 + s_zone] = win_1
    sig[-10 - s_zone:-10] += win_2
    return sig


def plot_sig(sig):
    plt.figure()
    plt.plot(sig)
    plt.grid()


def padding_level_signal():
    size_sig = 1000
    sig = create_pseudo_signal(size_sig)
    noise, sig = basis.add_noise(sig, 5)
    fft_sig = sf.rfft(sig)
    sig_aa = sf.irfft(fft_sig)
    sig_aa[100] = 40
    print(fft_sig.size)
    fft_sig_2 = sf.rfft(sig, size_sig * 2)
    print(fft_sig_2.size)
    #sig_2_aa = sf.irfft(fft_sig_2, size_sig)
    sig_2_aa = sf.irfft(fft_sig_2)[:size_sig]
    sig_2_aa[200] = 80
    plt.figure()
    plt.plot(sig, label='ori')
    plt.plot(sig_aa, label='aa')
    plt.plot(sig_2_aa, label='aa_2')
    plt.legend()
    plt.grid()    
    
    
def padding_level_signal_convol():
    size_sig = 1000
    t_sample = 100
    tau = 0.0005
    gain = 200
    # kernel
    ker = basis.kernel_exp(size_sig, t_sample, tau, gain)
    fft_ker = sf.rfft(ker)
    fft_ker_2 = sf.rfft(ker, size_sig * 2)
    plot_sig(ker)
    sig = create_pseudo_signal(size_sig)
    noise, sig = basis.add_noise(sig, 5)
    fft_sig = sf.rfft(sig)
    sig_aa = sf.ifftshift(sf.irfft(fft_sig*fft_ker))
    sig_aa[100] = 40
    print(fft_sig.size)
    fft_sig_2 = sf.rfft(sig, size_sig * 2)
    print(fft_sig_2.size)
    sig_2_aa = sf.irfft(fft_sig_2*fft_ker_2)[size_sig//2:-size_sig//2]
    sig_2_aa[200] = 80
    plt.figure()
    plt.plot(sig, label='ori')
    plt.plot(sig_aa,label='aa')
    plt.plot(sig_2_aa, label='aa_2')
    plt.legend()
    plt.grid()    


if __name__ == '__main__':
    # sig = create_pseudo_signal(200)
    # noise, sig = basis.add_noise(sig, 5)
    # plot_sig(sig)
    # basis.plot_dse(sig,"",17)
    padding_level_signal()
    #padding_level_signal_convol()
    plt.show()
