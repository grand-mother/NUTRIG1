"""
Created on 3 juil. 2023

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
from sradio.simu.du_resp import SimuDetectorUnitResponse
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


import proto.simu.proto_simu_du as simu
import proto.recons.proto_voc_efield as pvoc

#
# logger
#
logger = mlg.get_logger_for_script("recons")


BANDWIDTH = [31, 249]
BANDWIDTH = [52, 185]

def ref_efield(efield):
    # return t_max, e_max
    assert isinstance(efield, HandlingEfieldOfEvent)
    return efield.get_tmax_vmax()


def ref_efield_band(efield):
    # return t_max, e_max
    assert isinstance(efield, HandlingEfieldOfEvent)
    evt_band = efield.get_copy(efield.get_traces_passband(BANDWIDTH, False), True)
    assert isinstance(evt_band, HandlingEfieldOfEvent)
    evt_band.type_trace = f"E field {BANDWIDTH} MHz"
    t_m, e_m = evt_band.get_tmax_vmax(True, "parab")
    return t_m, e_m, evt_band


def ref_efield_band_causal(efield):
    # return t_max, e_max
    assert isinstance(efield, HandlingEfieldOfEvent)
    bandwich = BANDWIDTH
    evt_band = efield.get_copy(efield.get_traces_passband(bandwich, True), True)
    evt_band.type_trace = f"E Field {bandwich} MHz (causal)"
    t_m, e_m = evt_band.get_tmax_vmax(True, "parab")
    return t_m, e_m, evt_band


def ref_efield_no_noise(efield):
    # return t_max, e_max
    assert isinstance(efield, HandlingEfieldOfEvent)
    dus = SimuDetectorUnitResponse()
    dus.params.update(
        {
            "flag_add_leff": True,
            "flag_add_gal": False,
            "flag_add_rf": True,
            "fact_padding": 4,
            "lst": 18.0,
        }
    )
    dus.set_data_efield(efield)
    dus.set_xmax(efield.xmax)
    dus.compute_du_all()
    volt = efield.get_copy(dus.v_out, True)
    volt.set_unit_axis("$\mu$V", "dir", r"$V_{out}$") 
    volt.remove_traces_low_signal(600)
    volt.plot_footprint_val_max()
    ef_wnr = volt.get_copy(0)
    ef_wnr.type_trace = "E field Wiener 3D"
    ef_best = ef_wnr.get_copy(0)
    ef_best.type_trace = "E field Wiener BEST"
    assert isinstance(ef_wnr, HandlingEfieldOfEvent)
    v_apol = efield.get_polar_angle_efield()
    efield.network.plot_footprint_1d(np.rad2deg(v_apol),"Angle polar", scale="lin")
    tf_rf = dus.o_rfchain.get_tf_3d()
    dus.o_rfchain.plot_tf()
    #tf_rf[:,:] = 1
    r_leff = dus.o_ant3d.interp_leff.o_pre.range_itp
    r_leff = dus.o_ant3d.interp_leff.o_pre.get_idx_range(BANDWIDTH)
    fft_volt = sf.rfft(volt.traces, n=dus.size_with_pad)
    norm2_fft_volt = (fft_volt * np.conj(fft_volt)).real
    logger.info(norm2_fft_volt.shape)
    for idx_du in range(volt.get_nb_du()):
        dus.o_ant3d.set_name_pos(volt.idx2idt[idx_du], volt.network.du_pos[idx_du])
        dus.o_ant3d.interp_leff.set_angle_polar(v_apol[idx_du])
        ef_nn = np.zeros_like(fft_volt[idx_du])
        # SN
        tf_leff = dus.o_ant3d.interp_leff.get_fft_leff_pol(dus.o_ant3d.sn_leff)
        ef_nn[0][r_leff] = fft_volt[idx_du][0][r_leff] / ((tf_leff * tf_rf[0])[r_leff])
        # EW
        tf_leff = dus.o_ant3d.interp_leff.get_fft_leff_pol(dus.o_ant3d.ew_leff)
        ef_nn[1][r_leff] = fft_volt[idx_du][1][r_leff] / ((tf_leff * tf_rf[1])[r_leff])
        # UP
        tf_leff = dus.o_ant3d.interp_leff.get_fft_leff_pol(dus.o_ant3d.up_leff)
        ef_nn[2][r_leff] = fft_volt[idx_du][2][r_leff] / ((tf_leff * tf_rf[2])[r_leff])
        # best solution
        best_tfd_ef = pvoc.weight_efield_estimation(ef_nn, norm2_fft_volt[idx_du])
        # direct space
        ef_best.traces[idx_du][0] = sf.irfft(best_tfd_ef)[: dus.sig_size]
        ef_wnr.traces[idx_du] = sf.irfft(ef_nn)[:,: dus.sig_size]
    ef_wnr.get_tmax_vmax(True, "parab")
    ef_wnr.plot_footprint_val_max()
    ef_best.get_tmax_vmax(True, "parab")
    ef_best.plot_footprint_val_max()
    return 
    t_m, e_m = ef_wnr.get_tmax_vmax(False, "parab")
    return t_m, e_m, ef_wnr

def test_ref(f_efield):
    zh_f = ZhairesMaster(f_efield)
    efield = zh_f.get_object_3dtraces()
    s_info = zh_f.get_simu_info()
    efield.set_xmax(get_simu_xmax(s_info))
    t_m, e_m = ref_efield(efield)
    efield.plot_footprint_val_max()
    t_m1, e_m1, ef_1 = ref_efield_band(efield)
    ef_1.plot_footprint_val_max()
    t_m2, e_m2, ef_2 = ref_efield_band_causal(efield)
    ef_2.plot_footprint_val_max()
    ref_efield_no_noise(efield)
    # t_mw, e_mw, ef_w = ref_efield_no_noise(efield)
    # ef_w.plot_footprint_val_max()
    


def pipeline_01(f_efield):
    efield, volt = simu.proto_simu_vout(f_efield, "temp.asdf")


def recons_wiener(tr_volt):
    pass


if __name__ == "__main__":
    mlg.create_output_for_logger("debug")
    logger.info(mlg.string_begin_script())
    #
    f_efield = (
        "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
    )
    #
    test_ref(f_efield)
    #
    logger.info(mlg.string_end_script())
    plt.show()
