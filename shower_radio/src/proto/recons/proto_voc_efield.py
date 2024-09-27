"""
Created on 25 avr. 2023

@author: jcolley
"""

import pprint

import numpy as np
import scipy.fft as sf
import scipy.optimize as sco
import matplotlib.pyplot as plt

from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent
import sradio.io.sradio_asdf as fsr
from sradio.num.signal import WienerDeconvolutionWhiteNoise
import sradio.model.ant_resp as ant
from sradio.basis.frame import FrameDuFrameTan
from sradio.basis import coord
from sradio.num.signal import get_fastest_size_rfft
from sradio import set_path_model_du


from proto.simu.proto_simu_du import FILE_vout, FILE_efield



#
# logger
#
logger = mlg.get_logger_for_script("script")


#
# FILES
#
FILE_voc = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out_v_oc.asdf"
PATH_leff = "/home/jcolley/projet/grand_wk/data/model/detector"

set_path_model_du("/home/jcolley/projet/grand_wk/data/model")


def get_true_angle_polar(n_file, l_idt):
    print(n_file)
    f_zh = ZhairesMaster(n_file)
    evt = f_zh.get_object_3dtraces()
    evt.reduce_l_ident(l_idt)
    s_info = f_zh.get_simu_info()
    evt.set_xmax(get_simu_xmax(s_info))
    a_pol_rad = evt.get_polar_angle_efield(False)
    evt.network.plot_footprint_1d(
        np.rad2deg(a_pol_rad), "true angle polar", evt, scale="lin", unit="deg"
    )
    return a_pol_rad


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
    es_0 = wiener.get_spectrum_vec(trace[0])
    ar_es = np.empty((3, es_0.shape[0]), dtype=trace.dtype)
    ar_es[0] = es_0
    ar_es[1] = wiener.get_spectrum_vec(trace[1])
    ar_es[2] = wiener.get_spectrum_vec(trace[2])
    idx = np.argmax(np.mean(ar_es, axis=1))
    logger.debug(f"idx max energy spectrum  is {idx}")
    return ar_es[idx]


def weight_efield_estimation(tfd_ef, weight, plot=False):
    """

    :param tfd_ef:
    :type tfd_ef: float (3,n_f)
    :param weight:
    :type weight: float (3,n_f)
    """
    assert tfd_ef.shape == weight.shape
    l2_2 = False
    if l2_2:
        w_2 = weight*weight
        w_ef = np.sum(tfd_ef * w_2, axis=0)
        best_ef = w_ef / np.sum(w_2, axis=0)        
    else:
        w_ef = np.sum(tfd_ef * weight, axis=0)
        best_ef = w_ef / np.sum(weight, axis=0)
    if plot:
        plt.figure()
        plt.plot(np.abs(tfd_ef[0]), label="0")
        plt.plot(np.abs(tfd_ef[1]), label="1")
        plt.plot(np.abs(tfd_ef[2]), label="2")
        plt.plot(np.abs(best_ef), label="weight sol")
        plt.grid()
        plt.legend()
        plt.figure()
        plt.plot(weight[0], label="0")
        plt.plot(weight[1], label="1")
        plt.plot(weight[2], label="2")
        plt.grid()
        plt.legend()
    return best_ef


def check_recons_with_white_noise_ref():
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
    idx_du = 52
    sigma = 100
    # 1)
    evt, d_simu = fsr.load_asdf(FILE_voc)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    # 2)
    wiener = WienerDeconvolutionWhiteNoise(evt.f_samp_mhz * 1e-6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files(PATH_leff))
    evt.plot_footprint_val_max()
    evt.plot_trace_idx(idx_du)
    ##add white noise
    noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
    evt.traces[idx_du] += noise
    evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    ant3d.set_name_pos(evt.du_id[idx_du], evt.network.du_pos[idx_du])
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.4,
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
    wiener.set_spectrum_sig(es_sig_est)
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
    wiener.plot_spectrum()
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
    # wiener.plot_spectrum()
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
    idx_du = 25
    sigma = 10
    # 1)
    evt, d_simu = fsr.load_asdf(FILE_voc)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    # 2)
    wiener = WienerDeconvolutionWhiteNoise(evt.f_samp_mhz * 1e6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files(PATH_leff))
    # evt.plot_footprint_val_max()
    # evt.plot_trace_idx(idx_du)
    ## add white noise
    noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
    evt.traces[idx_du] += noise
    evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    ant3d.set_name_pos(evt.du_id[idx_du], evt.network.du_pos[idx_du])
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.4,
    )
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    ## compute polarization angle
    v_b = get_simu_magnetic_vector(d_simu)
    v_pol = np.cross(ant3d.cart_src_du, v_b)
    v_pol /= np.linalg.norm(v_pol)
    logger.info(f"vec pol: {v_pol}")
    # v_pol = np.array([0.8374681 , 0.32868347 ,0.43659407])
    # TEST: v_pol  and v_b must be orthogonal
    # assert np.allclose(np.dot(v_pol, v_b), 0)
    t_dutan = FrameDuFrameTan(ant3d.dir_src_du)
    v_pol_tan = t_dutan.vec_to(v_pol, "TAN")
    logger.info(v_pol_tan)
    # TEST: in TAN pol is in plane, test it
    # assert np.allclose(v_pol_tan[2], 0)
    logger.debug(v_pol_tan[2])
    polar = coord.tan_cart_to_polar_angle(v_pol_tan)
    polar = np.deg2rad(107)
    logger.debug(f"polar angle: {np.rad2deg(polar)}")
    ant3d.interp_leff.set_angle_polar(polar)
    ## get Leff for polar direction  and deconv
    # SN
    leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    wiener.set_rfft_kernel(leff_pol_sn)
    ## define energy spectrum of signal
    es_sig_est = get_max_energy_spectrum(evt.traces[idx_du], wiener)
    wiener.set_spectrum_sig(es_sig_est)
    ##
    sig, fft_sig_sn = wiener.deconv_white_noise(evt.traces[idx_du][0], sigma)
    # wiener.plot_measure_signal(" SN")
    sig_sn = sig[: evt.get_size_trace()]
    # EW
    leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    wiener.set_rfft_kernel(leff_pol_ew)
    fact = 1
    sig, fft_sig_ew = wiener.deconv_white_noise(evt.traces[idx_du][1], sigma * fact)
    # wiener.plot_measure_signal(f" fact {fact} to noise for EW")
    # wiener.plot_spectrum(False)
    wiener.plot_snr()
    sig_ew = sig[: evt.get_size_trace()]
    # UP
    leff_pol_up = ant3d.interp_leff.get_fft_leff_pol(ant3d.up_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    wiener.set_rfft_kernel(leff_pol_up)
    sig, fft_sig_up = wiener.deconv_white_noise(evt.traces[idx_du][2], sigma)
    # wiener.plot_measure_signal(" UP")
    # wiener.plot_spectrum()
    sig_up = sig[: evt.get_size_trace()]
    # best sol
    a_fft = np.array([fft_sig_sn, fft_sig_ew, fft_sig_up])
    sp_1 = wiener.get_spectrum_vec(evt.traces[idx_du, 0])
    sp_2 = wiener.get_spectrum_vec(evt.traces[idx_du, 1])
    sp_3 = wiener.get_spectrum_vec(evt.traces[idx_du, 2])
    # a_spec =  np.sqrt(np.array([sp_1, sp_2, sp_3]))
    a_spec = np.array([sp_1, sp_2, sp_3])
    best_fft_sig = weight_efield_estimation(a_fft, a_spec, True)
    best_sig = sf.irfft(best_fft_sig)[: evt.get_size_trace()]
    # plot sig estimation
    plt.figure()
    plt.title(f"E field polar estimation for each antenna\npolar angle: {np.rad2deg(polar):.1f}")
    plt.plot(evt.t_samples[idx_du], sig_sn, "k", label="E pol with SN data")
    plt.plot(evt.t_samples[idx_du], sig_ew, "y", label="E pol with EW data")
    plt.plot(evt.t_samples[idx_du], sig_up, "b", label="E pol with UP data")
    plt.plot(evt.t_samples[idx_du], best_sig, "g", label="E pol best")
    plt.ylabel(r"$\mu$V/m")
    plt.xlabel(f"ns")
    plt.grid()
    plt.legend()
    # evt.traces[idx_du] = np.array([sig_sn, sig_ew, sig_up])
    # evt.plot_trace_idx(idx_du)
    #
    if True:
        wiener.set_spectrum_sig(wiener.get_spectrum_vec(best_sig))
        # SN
        wiener.set_rfft_kernel(leff_pol_sn)
        sig, fft_sig_sn = wiener.deconv_white_noise(evt.traces[idx_du][0], sigma)
        wiener.plot_spectrum(False)
        wiener.plot_measure_signal("SN")
        # EW
        wiener.set_rfft_kernel(leff_pol_ew)
        sig, fft_sig_sn = wiener.deconv_white_noise(evt.traces[idx_du][1], sigma)
        # wiener.plot_measure_signal("EW")
        plt.plot(sig, label="EW")
        # UP
        wiener.set_rfft_kernel(leff_pol_up)
        sig, fft_sig_sn = wiener.deconv_white_noise(evt.traces[idx_du][2], sigma)
        # wiener.plot_measure_signal("UP")
        plt.plot(sig, label="UP")
        plt.legend()


def loss_func_polar(angle_pol, data):
    fft_volt = data[0]
    spect_volt = data[1]
    ant3d = data[2]
    wiener = data[3]
    sigma = data[4]
    coef_func2 = data[6]
    #
    ant3d.interp_leff.set_angle_polar(angle_pol)
    # SN
    leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    wiener.set_rfft_kernel(leff_pol_sn)
    _, fft_sig_sn = wiener.deconv_white_noise_fft_in(fft_volt[0], sigma)
    # wiener.plot_measure_signal(" SN")
    # EW
    leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    wiener.set_rfft_kernel(leff_pol_ew)
    _, fft_sig_ew = wiener.deconv_white_noise_fft_in(fft_volt[1], sigma)
    # wiener.plot_measure_signal(f" EW")
    # wiener.plot_spectrum(False)
    # UP
    leff_pol_up = ant3d.interp_leff.get_fft_leff_pol(ant3d.up_leff)
    wiener.set_rfft_kernel(leff_pol_up)
    _, fft_sig_up = wiener.deconv_white_noise_fft_in(fft_volt[2], sigma)
    # wiener.plot_measure_signal(" UP")
    # wiener.plot_spectrum()
    # best sol
    a_fft = np.array([fft_sig_sn, fft_sig_ew, fft_sig_up])
    # best_fft_sig = weight_efield_estimation(a_fft, spect_volt)
    best_fft_sig = np.sum(a_fft * spect_volt, axis=0) / np.sum(spect_volt, axis=0)
    data[5] = best_fft_sig
    # residu
    # r_sn_w = (fft_volt[0] - leff_pol_sn * best_fft_sig) * spect_volt[0]
    # r_ew_w = (fft_volt[1] - leff_pol_ew * best_fft_sig) * spect_volt[1]
    # r_up_w = (fft_volt[2] - leff_pol_up * best_fft_sig) * spect_volt[2]
    r_sn_w = fft_volt[0] - leff_pol_sn * best_fft_sig
    r_ew_w = fft_volt[1] - leff_pol_ew * best_fft_sig
    r_up_w = fft_volt[2] - leff_pol_up * best_fft_sig
    spect_volt_sum = np.sum(spect_volt, axis=0)
    residu = (
        r_sn_w * spect_volt[0] + r_ew_w * spect_volt[1] + r_up_w * spect_volt[2]
    ) / spect_volt_sum
    loss_func1 = np.sum((residu * np.conj(residu)).real)
    diff = r_sn_w - r_ew_w
    diff1 = ((diff * np.conj(diff)).real) * (spect_volt[0] + spect_volt[1])
    diff = r_sn_w - r_up_w
    diff2 = ((diff * np.conj(diff)).real) * (spect_volt[0] + spect_volt[2])
    diff = r_up_w - r_ew_w
    diff3 = ((diff * np.conj(diff)).real) * (spect_volt[2] + spect_volt[1])
    diff = (diff1 + diff2 + diff3) / (2 * spect_volt_sum)
    loss_func2 = np.sum(diff)
    loss_func = loss_func1 + coef_func2 * loss_func2
    logger.debug(f"for {np.rad2deg(angle_pol):5.1f} loss func: {loss_func:.1f}")
    return loss_func


def deconv_with_polar_fit_all_event(coef_func2=0.5, sigma=10):
    #
    evt, d_simu = fsr.load_asdf(FILE_voc)
    assert isinstance(evt, Handling3dTracesOfEvent)
    pprint.pprint(d_simu)
    ## pre-compute
    wiener = WienerDeconvolutionWhiteNoise(evt.f_samp_mhz * 1e6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files())
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.05,
    )
    logger.debug(freqs_out_mhz.shape)
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    a_pol = np.zeros(evt.get_nb_du(), dtype=np.float32)
    ## add white noise
    for idx_du in range(evt.get_nb_du()):
        noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
        evt.traces[idx_du] += noise
        # evt.plot_trace_idx(idx_du)
        # 2)
        ant3d.interp_leff.set_angle_polar(0)
        ant3d.set_name_pos(evt.idx2idt[idx_du], evt.network.du_pos[idx_du])
        ## define energy spectrum of signal
        leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
        wiener.set_rfft_kernel(leff_pol_sn)
        es_sig_est = get_max_energy_spectrum(evt.traces[idx_du], wiener)
        wiener.set_spectrum_sig(es_sig_est)
        #
        # minimize
        data = [1, 2, 3, 4, 5, 6, 7]
        # fft_volt
        v_0 = sf.rfft(evt.traces[idx_du][0], n=wiener.sig_size)
        v_1 = sf.rfft(evt.traces[idx_du][1], n=wiener.sig_size)
        v_2 = sf.rfft(evt.traces[idx_du][2], n=wiener.sig_size)
        data[0] = np.array([v_0, v_1, v_2])
        # spect_volt
        sp_1 = wiener.get_spectrum_vec(evt.traces[idx_du, 0])
        sp_2 = wiener.get_spectrum_vec(evt.traces[idx_du, 1])
        sp_3 = wiener.get_spectrum_vec(evt.traces[idx_du, 2])
        data[1] = np.array([sp_1, sp_2, sp_3])
        ## ant3d = data[2]
        data[2] = ant3d
        ## wiener = data[3]
        data[3] = wiener
        ## sigma = data[4]
        data[4] = sigma
        ## coef_func2, weight of second loss function
        if coef_func2 == "auto":
            max = np.max(np.abs(evt.traces[idx_du]))
            logger.info(f" max {max} of idx DU {idx_du}")
            snr = max ** 2 / sigma ** 2
            half_snr = 1 / (80 ** 2)
            coef_func2 = 1 / (1 + half_snr * snr ** 2)
        data[6] = coef_func2
        logger.info(mlg.chrono_start())
        res = sco.minimize_scalar(
            loss_func_polar, method="brent", args=data, tol=np.deg2rad(0.5), options={"disp": True}
        )
        logger.info(mlg.chrono_string_duration())
        logger.info(res.message)
        a_pol[idx_du] = np.rad2deg(res.x) % 180
        logger.info(a_pol[idx_du])
    # best_sig = sf.irfft(data[5])[: evt.get_size_trace()]
    # plt.figure()
    # plt.plot(evt.t_samples[idx_du], best_sig)
    # plt.grid()
    evt.network.plot_footprint_1d(a_pol, "fit polar angle", evt, scale="lin", unit="deg")
    return a_pol


def master_fit_polar():
    sigma = 25
    a_pol_0 = deconv_with_polar_fit_all_event(0, sigma)
    a_pol_1 = deconv_with_polar_fit_all_event(1, sigma)
    a_pol_2 = deconv_with_polar_fit_all_event(0.5, sigma)
    a_pol_3 = deconv_with_polar_fit_all_event("auto", sigma)
    a_pol = np.array([a_pol_0, a_pol_1, a_pol_2, a_pol_3])
    labels = [
        fr"||$\Delta$V||",
        "||$\Delta$V||+||$\Delta$E||",
        "||$\Delta$V||+||$\Delta$E||/2",
        "auto SNR",
    ]
    plt.figure()
    plt.title(f"Fit polar angle with voltage and Wierner estimator\n sigma noise {sigma} ")
    plt.boxplot(a_pol.T, showmeans=True, labels=labels)
    # plt.boxplot(a_pol_1 , autorange=True, showmeans=True)
    plt.ylabel("degree")
    plt.legend()
    plt.grid()


def loss_func_dir_polar(dir_pol, *data):
    fft_volt = data[0]
    spect_volt = data[1]
    ant3d = data[2]
    wiener = data[3]
    sigma = data[4]
    #
    ant3d.set_dir_source(dir_pol[:2])
    # ant3d.interp_leff.set_angle_polar(dir_pol[2])
    ant3d.interp_leff.set_angle_polar(np.deg2rad(116))
    # SN
    leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    wiener.set_rfft_kernel(leff_pol_sn)
    _, fft_sig_sn = wiener.deconv_white_noise_fft_in(fft_volt[0], sigma)
    # wiener.plot_measure_signal(" SN")
    # EW
    leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    wiener.set_rfft_kernel(leff_pol_ew)
    _, fft_sig_ew = wiener.deconv_white_noise_fft_in(fft_volt[1], sigma)
    # wiener.plot_measure_signal(f" EW")
    # wiener.plot_spectrum(False)
    # UP
    leff_pol_up = ant3d.interp_leff.get_fft_leff_pol(ant3d.up_leff)
    wiener.set_rfft_kernel(leff_pol_up)
    _, fft_sig_up = wiener.deconv_white_noise_fft_in(fft_volt[2], sigma)
    # wiener.plot_measure_signal(" UP")
    # wiener.plot_spectrum()
    # best sol
    a_fft = np.array([fft_sig_sn, fft_sig_ew, fft_sig_up])
    # best_fft_sig = weight_efield_estimation(a_fft, spect_volt)
    best_fft_sig = np.sum(a_fft * spect_volt, axis=0) / np.sum(spect_volt, axis=0)
    # data(5) = best_fft_sig
    # residu
    # r_sn_w = (fft_volt[0] - leff_pol_sn * best_fft_sig) * spect_volt[0]
    # r_ew_w = (fft_volt[1] - leff_pol_ew * best_fft_sig) * spect_volt[1]
    # r_up_w = (fft_volt[2] - leff_pol_up * best_fft_sig) * spect_volt[2]
    r_sn_w = fft_volt[0] - leff_pol_sn * best_fft_sig
    r_ew_w = fft_volt[1] - leff_pol_ew * best_fft_sig
    r_up_w = fft_volt[2] - leff_pol_up * best_fft_sig
    spect_volt_sum = np.sum(spect_volt, axis=0)
    residu = (
        r_sn_w * spect_volt[0] + r_ew_w * spect_volt[1] + r_up_w * spect_volt[2]
    ) / spect_volt_sum
    loss_func1 = np.sum((residu * np.conj(residu)).real)
    diff = r_sn_w - r_ew_w
    diff1 = ((diff * np.conj(diff)).real) * (spect_volt[0] + spect_volt[1])
    diff = r_sn_w - r_up_w
    diff2 = ((diff * np.conj(diff)).real) * (spect_volt[0] + spect_volt[2])
    diff = r_up_w - r_ew_w
    diff3 = ((diff * np.conj(diff)).real) * (spect_volt[2] + spect_volt[1])
    diff = (diff1 + diff2 + diff3) / (2 * spect_volt_sum)
    loss_func2 = np.sum(diff)
    loss_func = 0.01 * loss_func1 + loss_func2
    logger.debug(
        f"for {ant3d.interp_leff.dir_src_deg[0]:6.2f},{ant3d.interp_leff.dir_src_deg[1]:6.2f},{np.rad2deg(dir_pol[1]):6.2f} loss func: {loss_func:.1f} {np.log(loss_func)}"
    )
    return loss_func


def deconv_with_dir_polar_fit():
    idx_du = 52
    sigma = 2
    #
    evt, d_simu = fsr.load_asdf(FILE_voc)
    pprint.pprint(d_simu)
    ## add white noise
    noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
    evt.traces[idx_du] += noise
    evt.plot_trace_idx(idx_du)
    evt.plot_footprint_val_max()
    assert isinstance(evt, Handling3dTracesOfEvent)
    # 2)
    wiener = WienerDeconvolutionWhiteNoise(evt.f_samp_mhz * 1e6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files(PATH_leff))
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    true_dir = ant3d.interp_leff.dir_src_deg.copy()
    ant3d.interp_leff.set_angle_polar(0)
    ant3d.set_name_pos(evt.du_id[idx_du], evt.network.du_pos[idx_du])
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.20,
    )
    logger.debug(freqs_out_mhz.shape)
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    ## define energy spectrum of signal
    leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
    wiener.set_rfft_kernel(leff_pol_sn)
    es_sig_est = get_max_energy_spectrum(evt.traces[idx_du], wiener)
    wiener.set_spectrum_sig(es_sig_est)
    # GUESS
    # azimuth , dist zenith , polar
    guess = np.array([np.pi, np.pi / 4, np.pi / 2], dtype=np.float32)
    guess = np.array([290, 87.50429293, 116], dtype=np.float32)
    # guess += np.array([-30 , -40, -30])
    guess = np.deg2rad(guess)
    # minimize
    data = ["fft trace", 2, 3, 4, 5, 6, 7]
    # fft_volt
    v_0 = sf.rfft(evt.traces[idx_du][0], n=wiener.sig_size)
    v_1 = sf.rfft(evt.traces[idx_du][1], n=wiener.sig_size)
    v_2 = sf.rfft(evt.traces[idx_du][2], n=wiener.sig_size)
    data[0] = np.array([v_0, v_1, v_2])
    # spect_volt
    sp_1 = wiener.get_spectrum_vec(evt.traces[idx_du, 0])
    sp_2 = wiener.get_spectrum_vec(evt.traces[idx_du, 1])
    sp_3 = wiener.get_spectrum_vec(evt.traces[idx_du, 2])
    data[1] = np.array([sp_1, sp_2, sp_3])
    ## ant3d = data[2]
    data[2] = ant3d
    ## wiener = data[3]
    data[3] = wiener
    ## sigma = data[4]
    data[4] = sigma
    logger.info(mlg.chrono_start())
    if False:
        res = sco.minimize(
            loss_func_dir_polar,
            guess,
            bounds=([0, 0, 0], [2 * np.pi, np.pi / 2, np.pi]),
            method="BFGS",
            args=data,
            # xtol=np.deg2rad(0.5),
        )
        logger.info(mlg.chrono_string_duration())
        logger.info(res.message)
        logger.info(np.rad2deg(res.x))

    elif True:
        res = sco.least_squares(
            loss_func_dir_polar,
            guess[:2],
            # bounds=([0, 0, 0], [2*np.pi, np.pi / 2, np.pi]),
            bounds=([0, 0], [2 * np.pi, np.pi / 2]),
            method="dogbox",
            args=data,
            diff_step=np.deg2rad(5),
            x_scale=np.array([10, 1])
            # xtol=np.deg2rad(0.5),
        )
        logger.info(mlg.chrono_string_duration())
        logger.info(res.message)
        logger.info(np.rad2deg(res.x))
        # best_sig = sf.irfft(data[5])[: evt.get_size_trace()]
        # plt.figure()
        # plt.plot(evt.t_samples[idx_du], best_sig)
        # plt.grid()
    else:
        res = sco.shgo(
            loss_func_dir_polar,
            bounds=[(0, 2 * np.pi - 0.01), (0, np.pi / 2 - 0.01), (0, np.pi - 0.01)],
            args=data,
            n=2 ** 8,
        )
        logger.info(mlg.chrono_string_duration())
        logger.info(res.message)
        logger.info(np.rad2deg(res.x))
    logger.info(f"True dir : {true_dir}")


def check_recons_all_no_noise():
    f_plot_leff = False
    # 1)
    evt, d_simu = fsr.load_asdf(FILE_vout)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    evt.type_trace = "$V_{out}$"
    # evt.remove_traces_low_signal(5000)
    # evt.plot_ps_trace_idx(idx_du)
    # evt.plot_trace_idx(idx_du)
    #evt.downsize_sampling(4)
    evt_wnr = evt.get_copy(0)
    evt_wnr.type_trace = "E field wiener"
    evt.plot_footprint_val_max()
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files())
    # evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu["sim_shower"]))
    size_trace = evt.get_size_trace()
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        evt.get_size_trace(),
        evt.f_samp_mhz,
        1.4,
    )
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    ## compute polarization angle
    v_a_pol = get_true_angle_polar(d_simu["efield_file"], evt.idx2idt)
    for idx_du in range(evt.get_nb_du()):
        ant3d.set_name_pos(evt.idx2idt[idx_du], evt.network.du_pos[idx_du])
        ant3d.interp_leff.set_angle_polar(v_a_pol[idx_du])
        ## get Leff for polar direction  and deconv
        # EW
        leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
        ant3d.interp_leff.plot_leff_pol()
        # kernel  EW
        fft_tr = sf.rfft(evt.traces[idx_du][1], size_with_pad)
        r_leff = ant3d.interp_leff.o_pre.range_itp
        fft_sig_ew = np.zeros_like(fft_tr)
        fft_sig_ew[r_leff] =  fft_tr[r_leff]/ leff_pol_ew[r_leff]
        sig = sf.irfft(fft_sig_ew)
        evt_wnr.traces[idx_du][1] = sig[:size_trace]
    evt_wnr.get_tmax_vmax("parab")
    evt_wnr.plot_footprint_val_max()
    return evt_wnr


if __name__ == "__main__":
    mlg.create_output_for_logger("debug", log_root="script")
    logger.info(mlg.string_begin_script())
    # check_recons_with_white_noise()
    # deconv_with_polar_fit()
    # deconv_with_dir_polar_fit()
    deconv_with_polar_fit_all_event(1)
    #master_fit_polar()
    #check_recons_all_no_noise()
    #
    #
    logger.info(mlg.string_end_script())
    plt.show()
