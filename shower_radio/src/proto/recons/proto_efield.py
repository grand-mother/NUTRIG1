"""
Created on 25 avr. 2023

@author: jcolley
"""

import pprint

import numpy as np
import matplotlib.pylab as plt

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
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("debug")


#
# FILES
#
FILE_voc = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out_v_oc.asdf"
PATH_leff = "/home/jcolley/projet/grand_wk/data/model/detector"


def get_simu_magnetic_vector(d_simu):
    d_inc = d_simu["geo_mag2"]["inc"]
    r_inc = np.deg2rad(d_inc)
    v_b = np.array([np.cos(r_inc), 0, -np.sin(r_inc)])
    logger.info(f"Vec B: {v_b} , inc: {d_inc:.2f} deg")
    return v_b


def get_simu_xmax(d_simu):
    xmax = 1000.0 * np.array([d_simu["x_max"]["x"], d_simu["x_max"]["y"], d_simu["x_max"]["z"]])
    return xmax


def get_max_energy_spectrum(trace, wiener):
    es_0 = wiener.get_es_vec(trace[0])
    ar_es = np.empty((3, es_0.shape[0]), dtype=trace.dtype)
    ar_es[0] = es_0
    ar_es[1] = wiener.get_es_vec(trace[1])
    ar_es[2] = wiener.get_es_vec(trace[2])
    idx = np.argmax(np.mean(ar_es, axis=1))
    logger.debug(f"idx max energy spectrum  is {idx}")
    return ar_es[idx]


def check_recons_with_white_noise():
    """
    1) read v_oc file
    2) create wiener object
    3) on trace
        * add white noise
        * compute relative xmax and direction
        * compute polarization angle
        * get Leff for polar direction
        * deconv and store
    4) plot result
    5) estimate polarization or B orthogonality for all traces
    """
    f_plot_leff = False
    # 1)
    evt, d_simu = fsr.load_asdf(FILE_voc)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    # 2)
    wiener = WienerDeconvolution(evt.f_samp_mhz * 1e-6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files(PATH_leff))
    evt.plot_footprint_val_max()
    idx_du = 52
    evt.plot_trace_idx(idx_du)
    ##add white noise
    sigma = 50
    noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
    evt.traces[idx_du] += noise
    evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    ant3d.set_name_pos(evt.du_id[idx_du], evt.network.du_pos[idx_du])
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.2,
    )
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    ## compute polarization angle
    v_b = get_simu_magnetic_vector(d_simu)
    v_pol = np.cross(ant3d.cart_src_du, v_b)
    v_pol /= np.linalg.norm(v_pol)
    logger.info(f"vec pol: {v_pol}")
    # TEST: v_pol  and v_b must be orthogonal
    assert np.allclose(np.dot(v_pol, v_b), 0)
    t_dutan = FrameDuFrameTan(ant3d.dir_src_du)
    v_pol_tan = t_dutan.vec_to_b(v_pol)
    logger.info(v_pol_tan)
    # TEST: in TAN pol is in plane, test it
    assert np.allclose(v_pol_tan[2], 0)
    polar = coord.tan_cart_to_polar_angle(v_pol_tan)
    logger.debug(f"polar angle: {np.rad2deg(polar)}")
    ant3d.interp_leff.set_angle_polar(polar)
    ## get Leff for polar direction  and deconv
    # SN
    leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    wiener.set_rfft_kernel(leff_pol)
    ## define energy spectrum of signal
    es_sig_est = get_max_energy_spectrum(evt.traces[idx_du], wiener)
    wiener.set_es_sig(es_sig_est)
    ##
    sig = wiener.deconv_white_noise(evt.traces[idx_du][0], sigma)
    wiener.plot_measure_signal(" SN")
    sig_sn = sig[: evt.get_size_trace()]
    # EW
    leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    wiener.set_rfft_kernel(leff_pol)
    fact = 1
    sig = wiener.deconv_white_noise(evt.traces[idx_du][1], sigma * fact)
    wiener.plot_measure_signal(f" fact {fact} to noise for EW")
    wiener.plot_se()
    wiener.plot_snr()
    sig_ew = sig[: evt.get_size_trace()]
    # UP
    leff_pol = ant3d.interp_leff.get_fft_leff_pol(ant3d.up_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    wiener.set_rfft_kernel(leff_pol)
    sig = wiener.deconv_white_noise(evt.traces[idx_du][2], sigma)
    wiener.plot_measure_signal(" UP")
    # wiener.plot_se()
    sig_up = sig[: evt.get_size_trace()]
    # plot sig estimation
    plt.figure()
    plt.title("E field polar estimation for each antenna")
    plt.plot(evt.t_samples[idx_du], sig_sn, "k", label="E pol with SN data")
    plt.plot(evt.t_samples[idx_du], sig_ew, "y", label="E pol with EW data")
    plt.plot(evt.t_samples[idx_du], sig_up, "b", label="E pol with UP data")
    plt.ylabel(r"$\mu$V/m")
    plt.xlabel(f"ns")    
    plt.grid()
    plt.legend()
    evt.traces[idx_du] = np.array([sig_sn, sig_ew, sig_up])
    evt.plot_trace_idx(idx_du)


if __name__ == "__main__":
    check_recons_with_white_noise()
    plt.show()
