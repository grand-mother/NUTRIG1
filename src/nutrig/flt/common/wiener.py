'''
Created on 14 juin 2022

@author: jcolley
'''

import numpy as np
import scipy.signal as ss
import scipy.fft as fft
import matplotlib.pyplot as plt

from nutrig.flt.common.basis import kernel_exp, add_noise, Smoother

plt.ioff()

true_dse_s = 1

np.random.seed(12)


def create_sig_pseudo_dirac(s_sig, support_sample=5, level=1.0, pos=0.5):
    s_dirac = ss.hann(support_sample) * level
    idx_pos = int(s_sig * pos)
    sig = np.zeros(s_sig, dtype=np.float32)
    sig[idx_pos:idx_pos + support_sample] = s_dirac
    return sig

        
def create_sinc(s_sig, scale=1): 
    x = np.linspace(-20, 20, s_sig , endpoint=False)
    sig = np.sinc(x) * scale
    return sig * ss.hann(s_sig)


def convolve_by_exp_tau(sig):
    s_sig = sig.size
    ker = kernel_exp(s_sig, 0.01, 3)
    na_conv = ss.fftconvolve(sig, ker, 'same')
    plt.figure()
    plt.plot(sig, label='signal')
    plt.plot(na_conv, label='convolve')
    plt.grid()
    plt.legend()

    
def convolve_sig_by_kernel(sig, kernel): 
    # na_conv = ss.fftconvolve(sig, kernel, 'same')
    na_conv = np.real(fft.ifftshift((fft.ifft(fft.fft(sig) * fft.fft(kernel)))))
    return na_conv


def create_convolve_nois_signal(s_sig, sigma=0.2): 
    sig = 2 + create_sinc(s_sig, 3)
    # sig = create_sig_pseudo_dirac(s_sig, 40, 2)   
    ker = kernel_exp(s_sig, 0.01, 2)
    sig_conv = convolve_sig_by_kernel(sig, ker)    
    noise, sig_conv_noise = add_noise(sig_conv, sigma)
    if True:
        plt.figure()
        plt.title('Kernel')
        plt.plot(ker)
        plt.grid()
        plt.figure()
        plt.plot(sig, label='signal')
        plt.plot(sig_conv_noise, label='convol+noise')
        plt.grid()
        plt.legend()
        plot_dse(sig, "dse sig")
        plot_dse(noise, "noise")
        plot_dse(noise, "noise smooth", 30)    
    return sig, sig_conv_noise, ker 



def new_wiener_white_noise(measure, kernel, sigma):
    fft_m = fft.rfft(measure)
    fft_k = fft.rfft(kernel)
    se_k = (fft_k*np.conj(fft_k)).real
    se_s = (fft_m*np.conj(fft_m)).real- sigma
    idx_neg = np.where(se_s < 0)[0]
    se_s[idx_neg] = 0
    wiener = (np.conj(fft_k)*se_s)/(se_k*se_s + sigma)
    sol_w = fft.irfft(fft_m*wiener)
    return sol_w, wiener, se_s
    
    
    

def deconv_wiener_white_noise(measure, kernel, sigma):
    factor = np.sqrt(measure.size)
    # factor = 1
    print(factor)
    fft_k = fft.fft(kernel) / factor
    fft_k_conj = np.conj(fft_k)
    fft_m = fft.fft(measure) / factor
    # Wiener filter in fourier space
    dse_m = np.real(fft_m * np.conj(fft_m))
    dse_k = np.real(fft_k * fft_k_conj)
    var = sigma ** 2
    print("var: ", var)
    print('mean sig:' , np.sqrt(fft_m[0] / factor ** 2), np.mean(measure))
    if False:
        print("S : (dse_m - var)/fft_k")
        # fft_m[0] = 0+0j
        # smooth = Smoother(100, dse_m.size)
        # dse_smooth = smooth.hann(dse_m)
        snr_est = (dse_m - var)
        # snr_est = (fft_m - var)
        idx = np.where(snr_est < 0)[0]
        print("nb freq with val <0 : ", len(idx))
        # snr_est[idx] = 1e-10
        inv_snr_est = var * (np.real(fft_k) / factor) / snr_est
        # inv_snr_est = var*(dse_k) / snr_est

    if False:
        # like naive deconv
        print("S : (dse_m - var)/dse_k")
        # fft_m[0] = 0+0j
        # smooth = Smoother(100, dse_m.size)
        # dse_smooth = smooth.hann(dse_m)
        snr_est = (dse_m - var)
        snr_est[0] += var
        idx = np.where(snr_est < 0)[0]
        print("nb freq with val <0 : ", len(idx))
        snr_est[idx] = 1e-10
        inv_snr_est = (var * dse_k) / snr_est
        
    if True:
        idx_max = 30
        Q = 300
        
        Q_2_inv = 1 / (Q * Q)
        #
        inv_snr_est = Q_2_inv * np.ones(measure.size, dtype=np.float32)
        inv_snr_est[0] = 0
        inv_snr_est[1:idx_max + 1] = np.linspace(0, Q_2_inv, idx_max)
        #
        x_2 = np.arange(measure.size)
        x_2 = x_2 **5
        inv_snr_est = Q_2_inv * x_2 / (idx_max ** 5 + x_2)
        inv_snr_est[1] = 0
   
    if False:
        print("S : |(fft_m - var) / fft_k|**2")
        fft_s = (fft_m - var) / fft_k
        dse_s = np.real(np.abs(fft_s))
        dse_n = var
        inv_snr_est = (dse_n / dse_s)
        
    if False:
        print("With true DSE of signal")
        inv_snr_est = var / (true_dse_s * (factor ** 2))
        
    plt.figure()
    plt.title("inv SNR")
    # plt.yscale("log")
    plt.grid()
    plt.loglog(range(0, measure.size // 2), inv_snr_est[:measure.size // 2])
    # filter in FOURIER space
    # print("Naive déconv")
    # inv_snr_est = 0
    wiener_filter = fft_k_conj / (dse_k + inv_snr_est)
    # wiener_fourier = 1/fft_k
    deconv = np.real(fft.ifft(wiener_filter * fft_m))
    return fft.fftshift(deconv)


def perform_deconv_wiener():
    global true_dse_s
    s_sig = 2048
    sigma = 0.03
    sig, sig_conv_noise, ker = create_convolve_nois_signal(s_sig, sigma)
    # true dse of signal
    # true_dse_s = plot_dse(sig, "true DSE signal")
    # plot_dse(sig_conv_noise, "sig_conv_noise")
    plt.figure()   
    plt.plot(sig_conv_noise, label='measure=signal*ker+noise')
    plt.plot(sig, label='signal')    
    plt.grid()
    plt.legend()
    # deconv
    #deconv = deconv_wiener_white_noise(sig_conv_noise, ker, sigma)
    deconv, wiener, se_s = new_wiener(sig_conv_noise, ker, sigma)
    deconv = fft.ifftshift(deconv)
    smooth = Smoother(15, s_sig)
    plt.figure()   
    plt.plot(deconv, label='direct deconv solution')
    plt.plot(sig, label='signal')
    plt.plot(smooth.hann(deconv), '.--', label='direct deconv smooth')
    plt.grid()
    plt.legend()


# ## PLOT
def plot_kernel():
    s_sig = 512
    ker = kernel_exp(s_sig, 0.01, 2)    
    plt.figure()
    plt.grid()
    plt.plot(ker, ".")

    
def plot_create_sinc():
    s_sig = 2048
    na = create_sinc(s_sig, 3)
    plt.figure()
    plt.plot(na)

    
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
    # v_mean = np.median(dse_s_all)
    v_mean = dse_s.mean()
    print(v_mean)
    print(v_mean / sig.size)
    print(v_mean / np.sqrt(sig.size))
    plt.figure()
    plt.title(title + f' mean {mean_fft:.2} std {np.sqrt(v_mean):.2}')
    if smooth == 0:
        plt.loglog(freq[1:], dse_s)
    else:
        smooth = Smoother(smooth, dse_s.size)
        dse_smooth = smooth.hann(dse_s)
        plt.loglog(freq[1:], dse_smooth)
    plt.grid()
    return dse_s_all


# ## TEST
def test_smooth():
    size_sig = 1024
    x = np.linspace(-20, 20, size_sig)
    sig = np.sinc(x) * 10  
    sig_n = sig + np.random.normal(scale=0.01, size=size_sig)     
    smooth = Smoother(21, size_sig)
    plt.figure()
    plt.plot(sig, label="signal")
    plt.plot(sig_n, label="signal+noise")
    plt.plot(smooth.hann(sig_n), label="smoothed")
    plt.grid()
    plt.legend()

    
def test_add_noise():
    s_sig = 2048
    na = np.ones(s_sig, dtype=np.float32)
    plt.figure()
    plt.plot(na)    
    noise , na = add_noise(na, 5)
    plt.plot(na)


def test_dse():
    size_sig = 2048
    b_nor = 123 + np.random.normal(scale=2, size=size_sig)
    plot_dse(b_nor, "bruit gaussien")
    
    
def test_convol_causal():
    ""
    file = "/home/jcolley/projet/grand_wk/binder/xdu/Stshp_MZS_QGS204JET_Proton_3.98_79.6_90.0_9/a0.trace"
    a_trace = np.loadtxt(file)
    size_sample = (a_trace.shape[0]//2)*2
    size_signal = 50
    hann = ss.hann(size_signal)*10    
    #trace = np.zeros(size_sample, dtype=np.float32)
    #trace[500:500+size_signal] = hann
    trace = a_trace[:size_sample,1]
    ker = kernel_exp(size_sample,1,1/10.)
    conv_trace = convolve_sig_by_kernel(trace, ker)
    # ===== PLOT 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(f"convole ZHAires trace with causal kernel ")
    ax1.set_title("trace and convolution")
    ax1.plot(trace,label="signal")
    ax1.plot(conv_trace,label="signal convolved")
    ax1.grid()
    ax1.legend()
    ax2.set_title("kernel")
    ax2.plot(ker,label="causal kernel")
    ax2.grid()  
    plt.figure()
    plt.plot(trace,label="signal")
    plt.plot(conv_trace,label="convol with causal kernel")
    plt.legend()
    plt.grid()
    
if __name__ == "__main__":
    # plot_kernel()
    # plt.show()
    # exit(0)
    # test_smooth()
    # plot_create_sinc()
    # plot_kernel()
    s_sig = 2048
    # convolve_by_exp_tau(create_sinc(s_sig, 3))
    #create_convolve_nois_signal(s_sig)
    perform_deconv_wiener()
    #test_dse()
    plt.show()
