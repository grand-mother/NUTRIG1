"""
Created on 20 avr. 2023

@author: jcolley
"""

import numpy as np
import pprint
import copy

import matplotlib.pylab as plt
import scipy.fft as fft


from sradio.simu.du_resp import SimuDetectorUnitForEvent
from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent
import sradio.io.sradio_asdf as fsrad

G_path_leff = "/home/jcolley/projet/grand_wk/data/model/detector"
G_path_galaxy = ""
G_path_rf_chain = ""
G_path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
)
G_path_simu = "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"
# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)
logger.info("test")
f_out = "out_v_oc.asdf"


def wiener_white_noise(measure, kernel, sigma):
    fft_m = fft.rfft(measure)
    fft_k = fft.rfft(kernel)
    se_k = (fft_k * np.conj(fft_k)).real
    se_s = (fft_m * np.conj(fft_m)).real - sigma
    idx_neg = np.where(se_s < 0)[0]
    se_s[idx_neg] = 0
    wiener = (np.conj(fft_k) * se_s) / (se_k * se_s + sigma)
    sol_w = fft.irfft(fft_m * wiener)
    return sol_w, wiener, se_s


def proto_simu_voc():
    dus = SimuDetectorUnitForEvent(G_path_leff)
    event = ZhairesMaster(G_path_simu)
    data = event.get_object_3dtraces()
    d_info = event.get_simu_info()
    print(data)
    pprint.pprint(d_info)
    dus.set_data_efield(data)
    shower = {}
    shower["xmax"] = 1000 * np.array(
        [d_info["x_max"]["x"], d_info["x_max"]["y"], d_info["x_max"]["z"]]
    )
    dus.set_data_shower(shower)
    dus.compute_du_all()
    dus.o_ant3d.leff_ew.plot_leff_tan()
    dus.o_ant3d.leff_sn.plot_leff_tan()
    dus.o_ant3d.leff_up.plot_leff_tan()
    print(data.traces[0])
    print(dus.v_out[0])
    data.plot_footprint_val_max()
    assert isinstance(data, Handling3dTracesOfEvent)
    out = copy.copy(data)
    out.traces = dus.v_out
    out.set_unit_axis("$\mu$V", "dir")
    out.name += " V_oc"
    out.plot_footprint_val_max()
    fsrad.save_asdf_single_event(f_out, out, d_info)


def proto_read():
    event, info = fsrad.load_asdf(f_out)
    pprint.pprint(info)


if __name__ == "__main__":
    proto_simu_voc()
    proto_read()
    plt.show()
