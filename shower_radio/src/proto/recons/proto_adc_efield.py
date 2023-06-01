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
from sradio.num.signal import WienerDeconvolution
import sradio.model.ant_resp as ant
from sradio.basis.frame import FrameDuFrameTan
from sradio.basis import coord
from sradio.num.signal import get_fastest_size_rfft

#
# logger
#
logger = mlg.get_logger_for_script("script")
mlg.create_output_for_logger("debug", log_root="script")


#
# FILES
#
FILE_voc = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out_v_oc.asdf"
PATH_leff = "/home/jcolley/projet/grand_wk/data/model/detector"


def plot_dsx(trace, title="", f_samp_mhz=1, noverlap = 0, scaling="spectrum"):
    plt.figure()
    plt.title(f"Power spectrum {title}")
    freq, pxx_den = ssig.welch(
        trace,
        f_samp_mhz * 1e6,
        window="taylor",
        noverlap=noverlap,
        scaling=scaling,
    )
    plt.semilogy(freq[1:] * 1e-6, pxx_den[1:])
    # plt.plot(freq[2:] * 1e-6, pxx_den[2:], self._color[idx_axis], label=axis)
    plt.ylabel(f"((unit)^2")
    plt.xlabel(f"MHz")
    #plt.xlim([0, 400])
    plt.grid()
    plt.legend()



def study_noise_uniform():
    m_s = 2**21
    max_val = 10000
    n_bits = 14
    quantif = (2*max_val)/(2**n_bits)
    a_uni = np.random.uniform(-quantif/2,quantif/2, m_s)
    plt.figure()
    plt.plot(a_uni)
    plt.grid()
    plot_dsx(a_uni,"bruit uniforme", 2000, noverlap=1)
    plot_dsx(a_uni,"bruit uniforme", 2000, noverlap=10)
    plot_dsx(a_uni,"bruit uniforme", 2000, noverlap=100)
    
def study_noise_normal():
    m_s = 2**21
    max_val = 10000
    n_bits = 14
    quantif = (2*max_val)/(2**n_bits)
    a_uni = np.random.normal(0, quantif/2, m_s)
    print((quantif/2)**2)
    plt.figure()
    plt.plot(a_uni)
    plt.grid()
    plot_dsx(a_uni,"bruit normal", 2000, noverlap=1)
    plot_dsx(a_uni,"bruit normal", 2000, noverlap=10)
    plot_dsx(a_uni,"bruit normal", 2000, noverlap=100)
   
if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # 
    study_noise_uniform()
    study_noise_normal()
    # 
    logger.info(mlg.string_end_script())
    plt.show()
