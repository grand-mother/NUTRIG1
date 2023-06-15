"""
Created on 25 avr. 2023

@author: jcolley
"""

import pprint

import numpy as np
import scipy.fft as sf
import scipy.optimize as sco
import matplotlib.pylab as plt
import scipy.signal as ssig

import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent
import sradio.io.sradio_asdf as fsr
from sradio.num.signal import WienerDeconvolutionWhiteNoise
import sradio.model.ant_resp as ant
from sradio.basis.frame import FrameDuFrameTan
from sradio.basis import coord
from sradio.num.signal import get_fastest_size_rfft
from sradio import set_path_model_du
 

#
# logger
#
logger = mlg.get_logger_for_script("script")
mlg.create_output_for_logger("debug", log_root="script")


#
# FILES
#
set_path_model_du("/home/jcolley/projet/grand_wk/data/model")

S_STD = 20

def plot_se(trace, title=""):
    plt.figure()
    half = int(trace.shape[0]/2)
    fft_a = sf.rfft(trace)
    m_es = (np.conj(fft_a)*fft_a).real/trace.size
    #idx_neg = np.argwhere(m_es <=0)
    #m_es[idx_neg] = 1e-10
    mean = m_es.mean()
    plt.title(f"Energy Spectrum {title}, Mean :  {mean} ")
    plt.semilogy( m_es[2:-1])
    plt.ylabel(f"((unit)^2")
    plt.xlabel(f"MHz")
    plt.grid()
    

def plot_welch_density_wn(trace):
    logger.info("welch_density")
    plt.figure()
    nperseg = 1024
    m_window = ssig.get_window("flattop", nperseg)
    m_window /= m_window.sum()
    logger.info(f"my window {m_window.sum()}")
    fs=4
    freq, pxx_den = ssig.welch(
        trace,
        fs=fs,
        #window=np.ones(nperseg),
        # taylor
        #window=m_window,
        nperseg = nperseg,
        noverlap=None,
        nfft=None,
        detrend=False,
        scaling="density",
    )
    mean =  fs*pxx_den[2:-1].mean()/2
    plt.title(f"density, fs*Mean/2 :  {mean}")
    print((np.sqrt(mean)*np.sqrt(2)**2))
    plt.semilogy(freq[2:-1] * 1e-6, pxx_den[2:-1])
    plt.ylabel(f"((unit)^2")
    plt.xlabel(f"xHz")
    plt.grid()
    #plt.legend()
    #logger.info(f"{nperseg*np.mean(pxx_den[1:500])/2}")

def plot_welch_spectrum_wn(trace):
    """
    conclusion
    
    avec scaling="spectrum"
     * window est constante 1 ou n'importe quoi
         => nperseg*np.mean(pxx_den[1:-1])/2 est la variance
     * window est une fenetre qqconque d'aposidation normalise ou non
         => nperseg*np.mean(pxx_den[1:-1])/2 a un rseultat variable avec le type de fenetre !?
    
    """
    logger.info("welch_spectrum")
    plt.figure()
    nperseg = 512
    m_window = ssig.get_window("taylor", nperseg)
    m_window /= m_window.sum()
    logger.info(f"my window {m_window.sum()}")
    freq, pxx_den = ssig.welch(
        trace,
        fs=1,
        #window=np.ones(nperseg),
        # taylor
        window=m_window,
        nperseg = nperseg,
        noverlap=None,
        nfft=None,
        detrend=False,
        scaling="spectrum",
    )
    mean = nperseg*np.mean(pxx_den[1:-1])/2
    plt.title(f"spectrum, size_sig*Mean/2 :  {mean}")
    print((np.sqrt(mean)*np.sqrt(2)**2))
    plt.semilogy(freq[2:-1] * 1e-6, pxx_den[2:-1])
    plt.ylabel(f"((unit)^2")
    plt.xlabel(f"xHz")
    plt.grid()
    #plt.legend()
    logger.info(f"{nperseg*np.mean(pxx_den[1:500])/2}")


def plot_welch_dsx(trace, title="", f_samp_mhz=1, noverlap = 0, scaling="spectrum"):
    plt.figure()
    freq, pxx_den = ssig.welch(
        trace,
        #f_samp_mhz*1e6,
        #window="taylor",
        noverlap=noverlap,
        detrend=False,
        scaling=scaling,
    )
    mean = pxx_den[2:-1].mean()
    print(f"factor variance {(S_STD**2)/mean}")
    plt.title(f"Power spectrum {title} {scaling}, Mean :  {mean}")
    print((np.sqrt(mean)*np.sqrt(2)**2))
    plt.semilogy(freq[2:-1] * 1e-6, 85.2*pxx_den[2:-1])
    # plt.plot(freq[2:] * 1e-6, pxx_den[2:], self._color[idx_axis], label=axis)
    plt.ylabel(f"((unit)^2")
    plt.xlabel(f"MHz")
    #plt.xlim([0, 400])
    plt.grid()
    #plt.legend()

def plot_periodog_dsx(trace, title="", f_samp_mhz=1, scaling="spectrum"):
    plt.figure()
    freq, pxx_den = ssig.periodogram(
        trace,
        #f_samp_mhz*1e6,
        #window="taylor",
        detrend=False,
        scaling=scaling,
    )
    mean = pxx_den[2:-1].mean()
    print(f"factor variance {(S_STD**2)/mean}")
    plt.title(f"PERIODO : Power spectrum {title} {scaling}, Mean :  {mean}")
    print((np.sqrt(mean)*np.sqrt(2)**2))
    plt.semilogy(freq[2:-1] * 1e-6, 85.2*pxx_den[2:-1])
    # plt.plot(freq[2:] * 1e-6, pxx_den[2:], self._color[idx_axis], label=axis)
    plt.ylabel(f"((unit)^2")
    plt.xlabel(f"MHz")
    #plt.xlim([0, 400])
    plt.grid()
    #plt.legend()
    
     
 
def study_noise_normal_2():
    m_s = 2**20
    a_uni = np.random.normal(0, S_STD, m_s)
    print(f'Variance : {S_STD**2}')
    print(f'Check var: {a_uni.std()**2}')
    plot_welch_density_wn(a_uni)
    plot_welch_spectrum_wn(a_uni)
 
def study_noise_normal():
    m_s = 2**20
    a_uni = np.random.normal(0, S_STD, m_s)
    print(f'Variance : {S_STD**2}')
    print(f'Check var: {a_uni.std()**2}')
    plt.figure()
    plt.plot(a_uni)
    plt.grid()
    #plot_welch_dsx(a_uni,"bruit normal", 2000, noverlap=1)
    #plot_welch_dsx(a_uni,"bruit normal", 2000, noverlap=10)
    if False:
        plot_welch_dsx(a_uni,"bruit normal f=2000MHz", 2000, noverlap=100, scaling="density")
        plot_welch_dsx(a_uni,"bruit normal f=2000MHz", 2000, noverlap=100, scaling="spectrum")
        plot_welch_dsx(a_uni,"bruit normal f=500MHz", 500, noverlap=100, scaling="density")
        plot_welch_dsx(a_uni,"bruit normal f=500MHz", 500, noverlap=100, scaling="spectrum")
        plot_welch_dsx(a_uni,"bruit normal f=1MHz", 1, noverlap=100, scaling="density")
        plot_welch_dsx(a_uni,"bruit normal f=1MHz", 1, noverlap=100, scaling="spectrum")
    plot_welch_dsx(a_uni,"bruit normal f=1MHz", 100, noverlap=100, scaling="spectrum")
    half = int(m_s/4)
    print(type(half))
    plot_welch_dsx(a_uni[:half],"bruit normal f=1MHz", 100, noverlap=10, scaling="spectrum")
    plot_welch_dsx(a_uni[:half//2],"bruit normal f=1MHz", 10, noverlap=40, scaling="spectrum")
    plot_se(a_uni)
    plot_periodog_dsx(a_uni,"bruit normal f=100MHz", 100,  scaling="spectrum")
    plot_periodog_dsx(a_uni,"bruit normal f=1MHz", 1,  scaling="spectrum")
    
def define_ps_galrfchain():
    pass
    
    
if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # 
    #study_noise_uniform()
    study_noise_normal_2()
    # 
    logger.info(mlg.string_end_script())
    plt.show()
