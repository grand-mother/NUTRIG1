"""
Created on 20 avr. 2023

@author: jcolley
"""

import numpy as np
import pprint
import copy

import matplotlib.pylab as plt


from sradio.simu.du_resp import SimuDetectorUnitForEvent
from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg
from sradio.basis.traces_event import Handling3dTracesOfEvent


G_path_leff = "/home/jcolley/projet/grand_wk/data/model/detector"
G_path_galaxy = ""
G_path_rf_chain = ""
G_path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
)
G_path_simu = (
    "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"
)
# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)
logger.info("test")

def proto_simu_voc():
    dus = SimuDetectorUnitForEvent(G_path_leff)
    event = ZhairesMaster(G_path_simu)
    data = event.get_object_3dtraces()
    d_info = event.get_simu_info()
    print(data)
    pprint.pprint(d_info)
    dus.set_data_efield(data)
    shower = {}
    shower["xmax"] = 1000*np.array([d_info["x_max"]["x"], d_info["x_max"]["y"], d_info["x_max"]["z"]])
    dus.set_data_shower(shower)
    dus.compute_du_all()
    dus.o_ant3d.leff_ew.plot_leff_tan()
    dus.o_ant3d.leff_sn.plot_leff_tan()
    dus.o_ant3d.leff_up.plot_leff_tan()
    print(data.traces[0])
    print(dus.v_out[0])
    data.plot_footprint_val_max()
    out = copy.copy(data)
    out.traces = dus.v_out
    out.plot_footprint_val_max()


if __name__ == "__main__":
    proto_simu_voc()
    plt.show()