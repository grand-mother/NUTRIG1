"""
Created on 25 avr. 2023

@author: jcolley
"""

import pprint

import numpy as np
import scipy.fft as sf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import scipy.signal as ssig

from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent, get_psd
from sradio.basis.efield_event import HandlingEfieldOfEvent
import sradio.io.sradio_asdf as fsr
from sradio.io.shower.zhaires_base import get_simu_xmax, get_simu_magnetic_vector
from sradio.num.signal import WienerDeconvolution
from sradio.basis.frame import FrameDuFrameTan
from sradio.basis import coord
from sradio.num.signal import get_fastest_size_rfft
from sradio import set_path_model_du
from sradio.model.rf_chain import RfChainGP300
from sradio.model.galaxy import GalaxySignalGp300
import sradio.model.ant_resp as ant

from proto.recons.proto_voc_efield import get_max_energy_spectrum
from proto.simu.proto_simu_du import FILE_vout, FILE_efield
import proto.recons.proto_voc_efield as pvoc

#
# logger
#
logger = mlg.get_logger_for_script("vout_efield")


#
# FILES
#
set_path_model_du("/home/jcolley/projet/grand_wk/data/model")

# FILE_vout = "/home/jcolley/projet/grand_wk/data/volt/out4_v_out.asdf"
# FILE_vout = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out4_v_out.asdf"
# FILE_efield = (
#     "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
# )
# FILE_efield = (
#     "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
#     )

BANDWIDTH = [52, 185]


def check_galaxyGP300():
    logger.info("check_galaxyGP300")
    o_gal = GalaxySignalGp300()
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
        2000,
        2000,
        1.4,
    )
    for hour in range(2):
        fft_v_gal = o_gal.get_volt_all_du(18, size_with_pad, freqs_out_mhz, 10)
        o_gal.plot_v_ampl()
    fft_v_gal = o_gal.get_volt_all_du(18, size_with_pad, freqs_out_mhz, 10)
    plt.figure()
    n_0 = sf.irfft(fft_v_gal[0, 0])
    n_1 = sf.irfft(fft_v_gal[0, 1])
    n_2 = sf.irfft(fft_v_gal[0, 2])
    plt.plot(n_0, color="k")
    print(n_0.std(), n_1.std(), n_2.std())
    plt.plot(n_1, color="y")
    plt.plot(n_2, color="b")
    plt.grid()


def plot_psd_galrfchain(rfchain=None):
    f_sampl_mhz = 2000
    if rfchain is None:
        size_with_pad, freqs_out_mhz = get_fastest_size_rfft(
            200000,
            f_sampl_mhz,
            1.4,
        )
        rfchain = RfChainGP300()
        rfchain.compute_for_freqs(freqs_out_mhz)
    assert isinstance(rfchain, RfChainGP300)
    rfchain.plot_tf()
    o_gal = GalaxySignalGp300()
    size_with_pad = rfchain.lna.size_sig
    freqs_out_mhz = rfchain.lna.freqs_out
    fft_v_gal = o_gal.get_volt_all_du(18, size_with_pad, freqs_out_mhz, 10)
    tf_elec = rfchain.get_tf_3d()
    # fft_v_galelec[:,0] = fft_v_gal[:,0]*tf_elec[0]
    fft_v_galelec = fft_v_gal * tf_elec
    hdl_vgal = Handling3dTracesOfEvent("Simulation")
    hdl_vgal.set_unit_axis("$\mu$V", "dir", "$V_{gal}$")
    hdl_vgal.init_traces(sf.irfft(fft_v_gal), f_samp_mhz=rfchain.lna.f_samp_mhz)
    v_galelec = sf.irfft(fft_v_galelec[0])
    logger.info(v_galelec.shape)
    freq, pxx_den = get_psd(v_galelec, f_sampl_mhz, 512)
    logger.info(f"{freq.shape} {pxx_den.shape}")
    hdl_vgalelec = Handling3dTracesOfEvent("Simulation")
    hdl_vgalelec.set_unit_axis("$\mu$V", "dir", "$V_{gal}*IR_{elec}$")
    hdl_vgalelec.init_traces(sf.irfft(fft_v_galelec), f_samp_mhz=rfchain.lna.f_samp_mhz)
    idx = 0
    hdl_vgal.plot_trace_idx(idx)
    hdl_vgalelec.plot_trace_idx(idx)
    hdl_vgal.plot_ps_trace_idx(idx)
    hdl_vgalelec.plot_ps_trace_idx(idx)
    logger.info(hdl_vgal.welch_freq * 1e-6)
    logger.info(freqs_out_mhz)
    # freq, pxx_den = get_psd(sf.irfft(fft_v_galelec), freqs_out_mhz, 512)
    return freq * 1e-6, pxx_den


def define_psd_galrfchain(rfchain=None, lst=18):
    if rfchain is None:
        f_s = 2000
        size_with_pad, freqs_out_mhz = get_fastest_size_rfft(200000, f_s, 1.8)
        rfchain = RfChainGP300()
        rfchain.compute_for_freqs(freqs_out_mhz)
    assert isinstance(rfchain, RfChainGP300)
    rfchain.plot_tf()
    o_gal = GalaxySignalGp300()
    size_with_pad = rfchain.lna.size_sig
    freqs_out_mhz = rfchain.lna.freqs_out
    f_sampl_mhz = rfchain.lna.f_samp_mhz
    fft_v_gal = o_gal.get_volt_all_du(lst, size_with_pad, freqs_out_mhz, 1)
    tf_elec = rfchain.get_tf_3d()
    fft_v_galelec = fft_v_gal * tf_elec
    v_galelec = sf.irfft(fft_v_galelec)
    logger.info(v_galelec.shape)
    freq, pxx_den = get_psd(v_galelec[0], f_sampl_mhz, 250)
    logger.info(f"{freq.shape} {pxx_den.shape}")
    return freq, pxx_den


def check_recons_ew():
    """
    1) read v_out file
    2) create wiener object
    3) on trace
        * compute relative xmax and direction
        * compute polarization angle with Bxp
        * get Leff for polar direction
        * deconv and store
    4) plot result
    """
    f_plot_leff = False
    idx_du = 0
    # 1)
    evt, d_simu = fsr.load_asdf(FILE_vout)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    evt.type_trace = "$V_{out}$"
    # evt.plot_ps_trace_idx(idx_du)
    # evt.plot_trace_idx(idx_du)
    evt.downsize_sampling(4)
    evt.plot_ps_trace_idx(idx_du)
    evt.plot_trace_idx(idx_du)
    evt.plot_footprint_val_max()
    freq_noise, psd_galelc = define_psd_galrfchain()
    # 2)
    wiener = WienerDeconvolution(evt.f_samp_mhz * 1e6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files())
    # evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu))
    ant3d.set_name_pos(evt.idx2idt[idx_du], evt.network.du_pos[idx_du])
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
    logger.debug(f"polar angle: {np.rad2deg(polar)}")
    ant3d.interp_leff.set_angle_polar(polar)
    ## get Leff for polar direction  and deconv
    # EW
    leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
    if f_plot_leff:
        ant3d.interp_leff.plot_leff_tan()
        ant3d.interp_leff.plot_leff_pol()
    # kernel  EW
    rfchain = RfChainGP300()
    rfchain.compute_for_freqs(freqs_out_mhz)
    kernel = leff_pol_ew * rfchain.get_tf_3d()[1]
    wiener.set_rfft_kernel(kernel)
    # wiener.set_band(53, 190)
    ## define power spectrum density
    idx_max = np.argmax(np.abs(evt.traces[idx_du]).sum(axis=1))
    logger.debug(f"idx_max {idx_max}")
    freq_sig, psd_sig = get_psd(evt.traces[idx_du, idx_max], evt.f_samp_mhz, 147)
    plt.figure()
    plt.semilogy(freq_sig, psd_sig)
    psd_sig = wiener.get_interpol(freq_sig, psd_sig)
    plt.semilogy(wiener.a_freq_mhz, psd_sig)
    psd_galelc = wiener.get_interpol(freq_noise, psd_galelc[1])
    wiener.set_psd_noise(psd_galelc)
    sig, fft_sig_ew = wiener.deconv_measure(evt.traces[idx_du][1], psd_sig - psd_galelc)
    wiener.plot_measure_signal("EW")
    wiener.plot_psd(False)
    wiener.plot_snr()
    sig_ew = sig[: evt.get_size_trace()]
    # plot sig estimation
    plt.figure()
    plt.title(f"E field polar estimation for each antenna\npolar angle: {np.rad2deg(polar):.1f}")
    plt.plot(evt.t_samples[idx_du], sig_ew, "y", label="E pol with EW data")
    plt.ylabel(r"$\mu$V/m")
    plt.xlabel(f"ns")
    plt.grid()
    plt.legend()


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


def check_recons_with_ew():
    """
    1) read v_out file
    2) create wiener object
    3) on trace
        * compute relative xmax and direction
        * compute polarization angle with Bxp
        * get Leff for polar direction
        * deconv and store
    4) plot result
    """
    f_plot_leff = False

    # 1)
    evt, d_simu = fsr.load_asdf(FILE_vout)
    pprint.pprint(d_simu)
    assert isinstance(evt, Handling3dTracesOfEvent)
    evt.type_trace = "$V_{out}$"
    # evt.remove_traces_low_signal(5000)
    # evt.plot_ps_trace_idx(idx_du)
    # evt.plot_trace_idx(idx_du)
    # evt.downsize_sampling(4)
    ef_wnr = evt.get_copy(0)
    ef_wnr.type_trace = "E field wiener"
    evt.plot_footprint_val_max()
    freq_noise, psd_galelc = define_psd_galrfchain()
    # 2)
    wiener = WienerDeconvolution(evt.f_samp_mhz * 1e6)
    # 3)
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
    v_b = get_simu_magnetic_vector(d_simu["sim_shower"])
    v_pol = np.cross(ant3d.cart_src_du, v_b)
    v_pol /= np.linalg.norm(v_pol)
    logger.info(f"vec pol: {v_pol}")
    v_a_pol = get_true_angle_polar(d_simu["efield_file"], evt.idx2idt)
    print(v_a_pol)
    # v_pol = np.array([0.8374681 , 0.32868347 ,0.43659407])
    # TEST: v_pol  and v_b must be orthogonal
    # assert np.allclose(np.dot(v_pol, v_b), 0)
    rfchain = RfChainGP300()
    rfchain.compute_for_freqs(freqs_out_mhz)
    flag_psd_gal = True
    for idx_du in range(evt.get_nb_du()):
        ant3d.set_name_pos(evt.idx2idt[idx_du], evt.network.du_pos[idx_du])
        t_dutan = FrameDuFrameTan(ant3d.dir_src_du)
        v_pol_tan = t_dutan.vec_to(v_pol, "TAN")
        logger.info(v_pol_tan)
        # TEST: in TAN pol is in plane, test it
        # assert np.allclose(v_pol_tan[2], 0)
        logger.debug(v_pol_tan[2])
        polar = coord.tan_cart_to_polar_angle(v_pol_tan)
        logger.debug(f"polar angle: {np.rad2deg(polar)}")
        # ant3d.interp_leff.set_angle_polar(polar)
        ant3d.interp_leff.set_angle_polar(v_a_pol[idx_du])
        ## get Leff for polar direction  and deconv
        # EW
        leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
        # kernel  EW
        kernel = leff_pol_ew * rfchain.get_tf_3d()[1]
        wiener.set_rfft_kernel(kernel)
        ## define power spectrum density
        idx_max = np.argmax(np.abs(evt.traces[idx_du]).sum(axis=1))
        logger.debug(f"idx_max {idx_max}")
        freq_sig, psd_sig = get_psd(evt.traces[idx_du, idx_max], evt.f_samp_mhz, 147)
        psd_sig = wiener.get_interpol(freq_sig, psd_sig)
        if flag_psd_gal:
            print(freq_noise.shape, psd_galelc.shape)
            psd_galelc_w = wiener.get_interpol(freq_noise, psd_galelc[1])
            wiener.set_psd_noise(psd_galelc_w)
            flag_psd_gal = False
            wiener.set_band(BANDWIDTH)
        sig, fft_sig_ew = wiener.deconv_measure(evt.traces[idx_du][1], psd_sig - psd_galelc_w)
        if idx_du == 7:
            wiener.plot_psd(False)
            wiener.plot_snr()
        ef_wnr.traces[idx_du][1] = sig[:size_trace]
    # ef_wnr.plot_footprint_val_max()
    return ef_wnr


def check_recons_all():
    """
    Read voltage
    create efield Wiener container
    create object DetectorUnitAntenna3Axis
    get xmax position
    define size FFT
    compute angle of polarization for each DU
    define object rfchain
    get psd of noise for each axis, v_gal throught rf chain
    for each DU, voltage
        define psd estimation of signal
        set direction of xmax in DU frame
        set angle of polarization
        for each axis (3)
            get TF L_eff
            set total TF L_eff*rf chain
            set psd of noise
            get Wiener estimation of E field in bandwidth
        weight estimation of E field
    """
    f_plot_leff = False
    flag_true_polar = False
    # Read voltage
    volt, d_simu = fsr.load_asdf(FILE_vout)
    volt.set_periodogram(150)
    volt.remove_traces_low_signal(2000)
    t_max, v_max = volt.get_tmax_vmax()
    std_noise = 150
    snr_volt = v_max / std_noise
    logger.info(f"SNR {snr_volt}")
    pprint.pprint(d_simu)
    assert isinstance(volt, Handling3dTracesOfEvent)
    volt.type_trace = "$V_{out}$"
    # volt.reduce_l_ident(["A31", "A33", "A36"])
    # create efield Wiener container
    ef_wnr = HandlingEfieldOfEvent()
    ef_wnr.init_traces(np.zeros_like(volt.traces), volt.idx2idt, volt.t_start_ns, volt.f_samp_mhz)
    ef_wnr.init_network(volt.network.du_pos)
    ef_wnr.set_unit_axis(r"$\mu$V/m", "cart", "E Field Wiener")
    ef_best = ef_wnr.get_copy(0)
    ef_best.type_trace = "BEST Wiener"
    volt.plot_footprint_val_max()
    freq_noise, psd_galelc = define_psd_galrfchain()
    # 2)
    wiener = WienerDeconvolution(volt.f_samp_mhz * 1e6)
    # 3)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files())
    # evt.plot_trace_idx(idx_du)
    ## compute relative xmax and direction
    ant3d.set_pos_source(get_simu_xmax(d_simu["sim_shower"]))
    size_trace = volt.get_size_trace()
    size_with_pad, freqs_out_mhz = get_fastest_size_rfft(volt.get_size_trace(), volt.f_samp_mhz, 6)
    ant3d.set_freq_out_mhz(freqs_out_mhz)
    ## compute polarization angle
    if flag_true_polar:
        # True value
        v_apol = get_true_angle_polar(d_simu["efield_file"], volt.idx2idt)
    else:
        # guess geomagnetic polar angle
        v_apol = volt.network.get_polar_angle_geomagnetic(
            get_simu_magnetic_vector(d_simu["sim_shower"]), get_simu_xmax(d_simu["sim_shower"])
        )
    logger.info(f"vec pol: {v_apol}")
    #
    rfchain = RfChainGP300()
    rfchain.compute_for_freqs(freqs_out_mhz)
    tf_elec = rfchain.get_tf_3d()
    # init , strange ... but to improve
    wiener.set_rfft_kernel(tf_elec[0])
    wiener.set_band(BANDWIDTH)
    # interpol PSD galactic noise
    psd_galelc_sn = wiener.get_interpol(freq_noise, psd_galelc[0])
    psd_galelc_ew = wiener.get_interpol(freq_noise, psd_galelc[1])
    psd_galelc_up = wiener.get_interpol(freq_noise, psd_galelc[2])
    # fft of voltage and norm for weigth estimation
    fft_volt = sf.rfft(volt.traces, n=size_with_pad)
    norm2_fft_volt = (fft_volt * np.conj(fft_volt)).real
    for idx_du in range(volt.get_nb_du()):
        # if idx_du == 0:
        #     sig_noise_123 = np.std(volt.traces[idx_du][:,:100], axis=1)
        #     sig_noise_max = np.max(sig_noise_123)
        #     logger.info(f"std noise {sig_noise_max}")
        # define PSD of signal
        idx_max = np.argmax(np.abs(volt.traces[idx_du]).sum(axis=1))
        logger.debug(f"idx_max {idx_max}")
        freq_sig, psd_sig = get_psd(volt.traces[idx_du, idx_max], volt.f_samp_mhz)
        psd_sig = wiener.get_interpol(freq_sig, psd_sig)
        # define relative position of DU with Xmax and set polar angle
        ant3d.set_name_pos(volt.idx2idt[idx_du], volt.network.du_pos[idx_du])
        ant3d.interp_leff.set_angle_polar(v_apol[idx_du])
        ## get Leff , define TF and deconv
        ef_wiener = np.zeros_like(fft_volt[idx_du])
        # SN
        leff_pol_sn = ant3d.interp_leff.get_fft_leff_pol(ant3d.sn_leff)
        wiener.set_rfft_kernel(leff_pol_sn * tf_elec[0])
        wiener.set_psd_noise(psd_galelc_sn)
        sig1, ef_wiener[0] = wiener.deconv_fft_measure(fft_volt[idx_du][0], psd_sig - psd_galelc_sn)
        ef_wnr.traces[idx_du][0] = sig1[:size_trace]
        # EW
        leff_pol_ew = ant3d.interp_leff.get_fft_leff_pol(ant3d.ew_leff)
        wiener.set_rfft_kernel(leff_pol_ew * tf_elec[1])
        wiener.set_psd_noise(psd_galelc_ew)
        sig2, ef_wiener[1] = wiener.deconv_fft_measure(fft_volt[idx_du][1], psd_sig - psd_galelc_ew)
        ef_wnr.traces[idx_du][1] = sig2[:size_trace]
        # UP
        leff_pol_up = ant3d.interp_leff.get_fft_leff_pol(ant3d.up_leff)
        wiener.set_rfft_kernel(leff_pol_up * tf_elec[2])
        wiener.set_psd_noise(psd_galelc_up)
        sig3, ef_wiener[2] = wiener.deconv_fft_measure(fft_volt[idx_du][2], psd_sig - psd_galelc_up)
        ef_wnr.traces[idx_du][2] = sig3[:size_trace]
        # best Wiener solution
        best_tfd_ef = pvoc.weight_efield_estimation(ef_wiener, norm2_fft_volt[idx_du])
        # direct space
        ef_best.traces[idx_du][0] = sf.irfft(best_tfd_ef)[:size_trace]
        # if idx_du == 7:
        #     wiener.plot_psd(False)
        #     wiener.plot_snr()
    ef_wnr.plot_footprint_val_max()
    # ef_best.plot_footprint_val_max()
    return ef_best, snr_volt


def compare_efield(ef_wnr, snr_volt):
    zh_f = ZhairesMaster(FILE_efield)
    efield = zh_f.get_object_3dtraces()
    assert isinstance(efield, Handling3dTracesOfEvent)
    l_idx = [efield.idt2idx[idt] for idt in ef_wnr.idx2idt]
    logger.info(l_idx)
    efield.reduce_l_index(l_idx)
    evt_band = efield.get_copy(efield.get_traces_passband(BANDWIDTH))
    evt_band.type_trace = f"E field {BANDWIDTH} MHz"
    evt_band.plot_footprint_val_max()
    ef_wnr.plot_footprint_val_max()
    compare_evt(evt_band, ef_wnr, snr_volt)


def compare_evt(efield, wiener, snr_volt):
    assert isinstance(efield, Handling3dTracesOfEvent)
    assert isinstance(wiener, Handling3dTracesOfEvent)
    wiener_ok = wiener
    wiener.name = efield.name
    efield_ok = efield
    assert isinstance(efield_ok, Handling3dTracesOfEvent)
    assert isinstance(wiener_ok, Handling3dTracesOfEvent)
    tm_ef, em_ef = efield_ok.get_tmax_vmax(True, "parab")
    tm_v, em_v = wiener_ok.get_tmax_vmax(True, "parab")
    tm_diff = tm_v - tm_ef
    wiener_ok.network.plot_footprint_1d(
        tm_diff, "diff t_max (wiener - Efield band)", wiener_ok, "lin", "ns"
    )
    plt.figure()
    plt.title(f"diff t_max (wiener - Efield band)\n{efield.name}")
    plt.hist(tm_diff, bins=50)
    plt.xlabel("ns")
    plt.grid()
    em_diff = 100 * (em_ef - em_v) / em_ef
    wiener_ok.network.plot_footprint_1d(em_diff, "diff E_max ", wiener_ok, "lin", "%")
    print(em_diff)
    plt.figure()
    plt.title(
        f"Relative error of E_max (Wiener - Efield band)\nmean={em_diff.mean():.2f} std={em_diff.std():.2f}"
    )
    plt.hist(em_diff, bins=50)
    plt.xlabel(f"%\n{efield.name}")
    plt.grid()
    bbox_snr(em_diff, np.array(snr_volt), efield.name, "%", f"Relative error of Wiener Emax in {BANDWIDTH} MHz by bin of SNR")
    bbox_snr(tm_diff, np.array(snr_volt), efield.name, "ns", f"Error of Wiener Tmax in {BANDWIDTH} MHz by bin of SNR")

def bbox_snr(em_diff, snr_volt, f_name="", ylab="", m_title=""):
    l_snr = [50, 100, 200, 400]
    idx_last = len(l_snr) - 1
    l_er = []
    l_nb_er = []
    for idx, snr in enumerate(l_snr):
        if idx == 0:
            idx_ok = np.argwhere(snr_volt <= snr).ravel()
        else:
            idx_ok = np.argwhere(np.logical_and(snr_volt <= snr, snr_volt > l_snr[idx - 1])).ravel()
        l_er.append(em_diff[idx_ok])
        l_nb_er.append(len(l_er[-1]))
    idx_ok = np.argwhere(snr_volt >= l_snr[-1]).ravel()
    l_er.append(em_diff[idx_ok])
    l_nb_er.append(len(l_er[-1]))
    labels = [
        f"<50\n{l_nb_er[0]}",
        f"<100\n{l_nb_er[1]}",
        f"<200\n{l_nb_er[2]}",
        f"<400\n{l_nb_er[3]}",
        f">400\n{l_nb_er[4]}",
    ]
    plt.figure()
    plt.title(m_title)
    plt.boxplot(l_er, labels=labels, showfliers=False)
    plt.ylabel(ylab)
    plt.xlabel(f"SNR, nb DU in bin\n{f_name}")
    plt.legend()
    plt.grid()
    print(len(em_diff), len(snr_volt))


if __name__ == "__main__":
    mlg.create_output_for_logger("debug")
    logger.info(mlg.string_begin_script())
    #
    # check_recons_ew()
    # check_galaxyGP300()
    # define_psd_galrfchain()
    # plot_psd_galrfchain()
    # check_recons_ew()
    # ef_wnr = check_recons_with_ew()
    # compare_efield(ef_wnr)
    ef_wnr, snr = check_recons_all()
    compare_efield(ef_wnr, snr)
    #
    logger.info(mlg.string_end_script())
    plt.show()
