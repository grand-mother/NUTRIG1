
import numpy as np
import scipy.signal as ss
import scipy.fft as fft
import matplotlib.pyplot as plt


def kernel_exp(s_sig, t_sample, tau=1, gain=3):
    x = np.linspace(0, t_sample * s_sig, s_sig // 2 , endpoint=False)
    ker = np.zeros(s_sig, dtype=np.float32)
    ker[s_sig // 2:] = gain * np.exp(-tau * x)    
    ker /= ker.sum()
    return ker


def add_noise(a_sig, sigma):
    print(a_sig.shape)
    noise = np.random.normal(scale=sigma, size=a_sig.size)
    return noise, a_sig + noise



class Smoother(object):

    def __init__(self, size_window=5, size_sig=2048):
        hann = ss.hann(size_window)        
        ker_hann = np.zeros(size_sig, dtype=np.float32)
        half_s = size_window // 2
        print(size_window, half_s)
        if (size_window % 2) == 0:
            ker_hann[0:half_s] = hann[half_s:]
        else:
            ker_hann[0:half_s + 1] = hann[half_s:]
        ker_hann[-half_s:] = hann[0:half_s]
        print(hann.sum())
        print(ker_hann.sum())
        self.ker_hann = ker_hann / ker_hann.sum()        
        self.fft_ker = fft.fft(self.ker_hann)
    
    def hann(self, sig):
        fft_sig = fft.fft(sig)
        return np.real(fft.ifft(fft_sig * self.fft_ker))



def plot_dse(sig, title="", smooth=0):
    """
    fft par defaut:
      * DSE = fft*fft_conj/sig.size
      * mean => sqrt(DSE[0] /sig.size) (imaginaire nulle)
      * var  => mean(DSE[1:])
    """
    freq = np.arange(sig.size // 2)
    print('mean sig : ', np.mean(sig))
    print("var sig : ", np.std(sig) ** 2)
    fft_m = fft.fft(sig)
    dse_s_all = np.real(fft_m * np.conj(fft_m)) / sig.size
    mean_fft = np.sqrt(dse_s_all[0] / sig.size)
    print('DSE[0]=', mean_fft)
    print('DSE[1]=', dse_s_all[1])
    print('DSE[Nyquist]=', dse_s_all[sig.size // 2])
    idx = 10
    print(f'DSE[{idx}]={dse_s_all[idx]} DSE[{sig.size- idx}]={dse_s_all[sig.size- idx]}')
    dse_s = dse_s_all[1:sig.size // 2] 
    v_mean = np.median(dse_s_all)
    #v_mean = dse_s.mean()
    print(v_mean)
    print(v_mean / sig.size)
    print(v_mean / np.sqrt(sig.size))
    plt.figure()
    plt.title(title + f' mean {mean_fft:.2} std~{np.sqrt(v_mean):.2}')
    if smooth == 0:
        plt.loglog(freq[1:], dse_s)
    else:
        smooth = Smoother(smooth, dse_s.size)
        dse_smooth = smooth.hann(dse_s)
        plt.loglog(freq[1:], dse_smooth)
    plt.grid()
    return dse_s_all
