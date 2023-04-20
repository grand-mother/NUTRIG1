"""
Created on 20 avr. 2023

@author: jcolley
"""

from sradio.simu.du_resp import SimuDetectorUnitForEvent
from sradio.io.shower.zhaires_master import ZhairesMaster

G_path_leff = "/home/jcolley/projet/grand_wk/data/model/detector"
G_path_galaxy = ""
G_path_rf_chain = ""
G_path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"

def proto_simu_voc():
    dus = SimuDetectorUnitForEvent(G_path_leff)
    event = ZhairesMaster(G_path_simu)
    data = event.get_object_3dtraces()
    d_info = event.get_simu_info()
    print(data)
    print(d_info)
    dus.set_data_efield(data)

if __name__ == "__main__":
    proto_simu_voc()