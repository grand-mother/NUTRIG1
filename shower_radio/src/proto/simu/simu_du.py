"""
Created on 20 avr. 2023

@author: jcolley
"""
import pprint
import copy

import numpy as np
import scipy.fft as sf
import matplotlib.pylab as plt

from sradio.simu.du_resp import SimuDetectorUnitResponse
from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent
from sradio.basis.efield_event import efield_in_polar_frame
import sradio.io.sradio_asdf as fsrad
from sradio.io.shower import zhaires_base as zbase
from sradio.basis.frame import FrameDuFrameTan
from sradio.basis import coord
import sradio.num.signal as srs


#
# Path file
#
G_path_leff = "/home/jcolley/projet/grand_wk/data/model/detector"
G_path_galaxy = ""
G_path_rf_chain = ""
# G_path_simu = (
#     "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
# )
# G_path_simu = "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"
G_path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
)
G_Voc_out = "out_v_oc.asdf"

#
# Logger
#
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("debug", log_stdout=True)


def view_efield_passband(f_simu, idx):
    f_zh = ZhairesMaster(f_simu)
    evt = f_zh.get_object_3dtraces()
    evt.plot_trace_idx(idx)
    tr_band = srs.filter_butter_band_fft(evt.traces[idx][0], 50*1e-6, 230*1e-6, 1e-6*evt.f_samp_mhz)
    evt.traces[idx][0] = tr_band
    tr_band = srs.filter_butter_band_fft(evt.traces[idx][1], 50*1e-6, 230*1e-6, 1e-6*evt.f_samp_mhz)
    evt.traces[idx][1] = tr_band
    tr_band = srs.filter_butter_band_fft(evt.traces[idx][2], 50*1e-6, 230*1e-6, 1e-6*evt.f_samp_mhz)
    evt.traces[idx][2] = tr_band
    evt.plot_trace_idx(idx)

def view_efield_polar_passband(f_simu, idx):
    f_zh = ZhairesMaster(f_simu)
    evt = f_zh.get_object_3dtraces()
    evt.plot_trace_idx(idx)
    efield1d, pol_est = efield_in_polar_frame(evt.traces[idx])
    tr_band = srs.filter_butter_band_fft(efield1d, 50*1e-6, 230*1e-6, 1e-6*evt.f_samp_mhz)
    plt.figure()
    plt.plot(evt.t_samples[idx], efield1d, label="E polar")
    plt.legend()
    plt.figure()
    plt.plot(evt.t_samples[idx], tr_band, label="E polar antenna bandwidth")
    plt.legend()
    plt.grid()



def estimate_polar_vec(trace, threasold=20):
    """

    :param efield_3d: (n_s,3)
    """
    assert trace.shape[1] == 3
    print("===========estimate_polar_vec================")
    n_elec = np.linalg.norm(trace, axis=1)
    idx_hb = np.where(n_elec > threasold)[0]
    # unit
    sple_ok = trace[idx_hb].T / n_elec[idx_hb]
    idx_neg = np.where(sple_ok[1] < 0)[0]
    sple_ok = sple_ok.T
    sple_ok[idx_neg] = -sple_ok[idx_neg]
    n_elec_2 = n_elec[idx_hb] * n_elec[idx_hb]
    temp = sple_ok.T * n_elec_2
    logger.info(temp.shape)
    pol_est = np.sum(temp, axis=1) / np.sum(n_elec_2)
    pol_est /= np.linalg.norm(pol_est)
    logger.info(pol_est.shape)
    logger.info(pol_est)
    logger.info(np.linalg.norm(pol_est))
    return pol_est


def test_simu_in_frame_pol(f_simu):
    dus = SimuDetectorUnitResponse(G_path_leff)
    zmf = ZhairesMaster(f_simu)
    evt = zmf.get_object_3dtraces()
    d_simu = zmf.get_simu_info()
    pprint.pprint(d_simu)
    # idx_du = 124, 52; 123
    idx_du = 52
    evt.plot_footprint_val_max()
    sigma = 20
    noise = np.random.normal(0, sigma, (3, evt.get_size_trace()))
    evt.traces[idx_du] += noise
    evt.plot_trace_idx(idx_du)
    xmax = zbase.get_simu_xmax(d_simu)
    logger.info(xmax)
    dus.set_data_efield(evt)
    dus.set_xmax(xmax)
    # compute Ref in [DU] frame
    dus.compute_du_idx(idx_du)
    # Now in polar frame
    t_dutan = FrameDuFrameTan(dus.o_ant3d.dir_src_du)
    ef_tan = t_dutan.vec_to_b(evt.traces[idx_du])
    idx_max = np.argmax(evt.traces[idx_du][1])
    logger.info(idx_max)
    logger.info(ef_tan.shape)
    logger.info(ef_tan[:, idx_max - 2 : idx_max + 2])
    # compute E fied polar
    pol_est = estimate_polar_vec(evt.traces[idx_du].T)
    efield_pol = np.dot(evt.traces[idx_du].T, pol_est)
    fft_efield_pol = sf.rfft(efield_pol, n=dus.size_with_pad)
    plt.plot(evt.t_samples[idx_du], efield_pol, label="E polar")
    plt.legend()
    # compute polar angle
    pol_est_tan = t_dutan.vec_to_b(pol_est)
    angle_pol = coord.tan_cart_to_polar_angle(pol_est_tan)
    logger.info(f"Angle polar: {np.rad2deg(angle_pol):.1f}")
    dus.o_ant3d.interp_leff.set_angle_polar(angle_pol)
    fft_v_oc = dus.o_ant3d.get_resp_1d_efield_pol(fft_efield_pol)
    v_oc = sf.irfft(fft_v_oc)[:, : dus.sig_size]
    v_oc_pol = v_oc
    plt.figure()
    plt.title("V_oc polar")
    plt.plot(v_oc[0], "k", label="SN")
    plt.plot(v_oc[1], "y", label="EW")
    plt.plot(v_oc[2], "b", label="UP")
    plt.legend()
    plt.grid()
    plt.figure()
    plt.title("V_oc DU")
    plt.plot(dus.v_out[idx_du][0], "k", label="SN")
    plt.plot(dus.v_out[idx_du][1], "y", label="EW")
    plt.plot(dus.v_out[idx_du][2], "b", label="UP")
    plt.legend()
    plt.grid()
    #
    plt.figure()
    plt.title("V_oc SN")
    plt.plot(v_oc[0], label="polar")
    plt.plot(dus.v_out[idx_du][0], label="DU")
    plt.legend()
    plt.grid()
    #
    plt.figure()
    plt.title("V_oc EW")
    plt.plot(v_oc[1], label="polar")
    plt.plot(dus.v_out[idx_du][1], label="DU")
    plt.legend()
    plt.grid()
    #
    plt.figure()
    plt.title("V_oc UP")
    plt.plot(v_oc[2], label="polar")
    plt.plot(dus.v_out[idx_du][2], label="DU")
    plt.legend()
    plt.grid()
    return v_oc_pol


def proto_simu_voc(f_out=None):
    dus = SimuDetectorUnitResponse(G_path_leff)
    event = ZhairesMaster(G_path_simu)
    data = event.get_object_3dtraces()
    d_info = event.get_simu_info()
    print(data)
    pprint.pprint(d_info)
    dus.set_data_efield(data)
    shower = {}
    shower["xmax"] = zbase.get_simu_xmax(d_info)
    dus.set_data_shower(shower)
    dus.compute_du_all()
    dus.o_ant3d.interp_leff.plot_leff_tan()
    dus.o_ant3d.interp_leff.get_fft_leff_du(dus.o_ant3d.sn_leff)
    print(data.traces[0])
    print(dus.v_out[0])
    data.plot_footprint_val_max()
    assert isinstance(data, Handling3dTracesOfEvent)
    out = copy.copy(data)
    out.traces = dus.v_out
    out.set_unit_axis("$\mu$V", "dir")
    out.name += "V_oc"
    out.plot_footprint_val_max()
    if f_out:
        fsrad.save_asdf_single_event(f_out, out, d_info)


def proto_read():
    event, info = fsrad.load_asdf(G_Voc_out)
    pprint.pprint(info)


if __name__ == "__main__":
    #proto_simu_voc()
    # proto_read()
    #test_simu_in_frame_pol(G_path_simu)
    #view_efield_passband("/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2", 52)
    view_efield_polar_passband(
        "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2",
        52)
    plt.show()
