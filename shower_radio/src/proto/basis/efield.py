"""
Created on 15 mai 2023

@author: jcolley
"""
import pprint
import matplotlib.pyplot as plt
from sradio.io.shower.zhaires_master import ZhairesMaster
from sradio.io.shower.zhaires_base import get_simu_xmax, get_simu_magnetic_vector
import sradio.manage_log as mlg
import sradio.basis.coord as coord
from sradio.basis.efield_event import HandlingEfieldOfEvent
from sradio.basis.frame import FrameDuFrameTan
import numpy as np
import sradio.model.ant_resp as ant

PATH_leff = "/home/jcolley/projet/grand_wk/data/model/detector"
G_path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
)
# G_path_simu = "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"
G_path_simu = (
    "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
)
# G_path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_LH_EPLHC_Proton_3.98_84.5_180.0_2"
#
# Logger
#
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger("debug", log_stdout=True)


def get_polar_angle_by_efield(f_efield):
    f_zh = ZhairesMaster(f_efield)
    i_sim = f_zh.get_simu_info()
    pprint.pprint(f_zh.get_simu_info())
    evt = f_zh.get_object_3dtraces()
    a_pol_du = evt.get_polar_vec(threshold=20)
    ant3d = ant.DetectorUnitAntenna3Axis(ant.get_leff_from_files(PATH_leff))
    ant3d.set_pos_source(get_simu_xmax(i_sim))
    a_pol = np.zeros(evt.get_nb_du(), dtype=np.float32)
    for idx_du in range(evt.get_nb_du()):
        ant3d.set_name_pos(evt.idx2idt[idx_du], evt.network.du_pos[idx_du])
        t_dutan = FrameDuFrameTan(ant3d.dir_src_du)
        v_pol_tan = t_dutan.vec_to(a_pol_du[idx_du], "TAN")
        a_pol[idx_du] = np.rad2deg(coord.tan_cart_to_polar_angle(v_pol_tan))
    evt.network.plot_footprint_1d(
        a_pol, "fit polar angle from Efield", evt, scale="lin", unit="deg"
    )
    evt.plot_footprint_val_max()
    evt_filter = evt.get_copy(evt.get_traces_passband())
    evt_filter.plot_footprint_val_max()


def test_fit_polar(f_simu):
    f_zh = ZhairesMaster(f_simu)
    i_sim = f_zh.get_simu_info()
    pprint.pprint(f_zh.get_simu_info())
    evt = f_zh.get_object_3dtraces()
    # evt.network.name += f"\nXmax dist {i_sim['x_max']['dist']:.1f}km, zenith angle: {i_sim['shower_zenith']:.1f}deg"
    evt.plot_footprint_val_max()
    # a_pol  = evt.get_polar_vec()
    # print(a_pol)
    evt.plot_polar_check_fit()


def compare_polar_angle(f_simu):
    f_zh = ZhairesMaster(f_simu)
    i_sim = f_zh.get_simu_info()
    pprint.pprint(f_zh.get_simu_info())
    evt = f_zh.get_object_3dtraces()
    evt.set_xmax(get_simu_xmax(i_sim))
    mfield = get_simu_magnetic_vector(i_sim)
    pol_ef_deg = evt.get_polar_angle_efield(degree=True)
    pol_mf_deg = evt.network.get_polar_angle_geomagnetic(mfield, evt.xmax,degree=True)
    # print(pol_ef)
    evt.network.plot_footprint_1d(
        pol_ef_deg, "Polar angle with E field (True)", evt, scale="lin", unit="deg"
    )
    evt.network.plot_footprint_1d(pol_mf_deg, "Polar angle geomagnetic", evt, scale="lin", unit="deg")
    evt.network.plot_footprint_1d(
        pol_mf_deg - pol_ef_deg, "Error polar angle (geomagnetic-True)", evt, scale="lin", unit="deg"
    )


def test_filter(f_simu):
    f_zh = ZhairesMaster(f_simu)
    i_sim = f_zh.get_simu_info()
    pprint.pprint(f_zh.get_simu_info())
    evt = f_zh.get_object_3dtraces()
    assert isinstance(evt, HandlingEfieldOfEvent)
    evt_band = evt.get_copy(evt.get_traces_passband([50, 200]))
    evt_band.type_trace = "E field [50,200]MHz"
    evt_band.plot_footprint_val_max()
    evt.plot_footprint_val_max()


if __name__ == "__main__":
    # test_fit_polar(G_path_simu)
    # get_polar_angle_by_efield(G_path_simu)
    # test_filter(G_path_simu)
    compare_polar_angle(G_path_simu)
    plt.show()
