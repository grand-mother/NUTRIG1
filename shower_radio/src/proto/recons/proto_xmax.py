"""

"""
import numpy as np
import matplotlib.pyplot as plt
import pprint

from sradio.basis.du_network import DetectorUnitNetwork
import sradio.recons.shower_plane as pwf
import sradio.recons.shower_spheric as swf
from sradio.io.shower.zhaires_master import ZhairesMaster

MY_PATH_DATA = "/home/jcolley/projet/grand_wk/dc1/grand/data"

REF_POS = MY_PATH_DATA+"/tests/recons/ref_recons_coord_antennas.txt"
REF_EVENT = MY_PATH_DATA+"/tests/recons/ref_recons_coinctable.txt"

G_file_efield = "/home/dc1/Coarse2_xmax_add.root"
G_file_efield = "/home/dc1/Coarse3.root"
G_file_efield = "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"


def read_ref_data():
    # assume REF_POS and REF_TIME in same order
    # read only x, y,z
    pos = np.loadtxt(REF_POS, usecols=(1, 2, 3))
    event = np.loadtxt(REF_EVENT, usecols=(2, 3))
    t_evt = event[:, 0]
    val_evt = event[:, 1]
    print(pos.shape, t_evt.shape, val_evt.shape)
    return pos, t_evt, val_evt


def plot_event(pos, t_evt, val_evt):
    net = DetectorUnitNetwork()
    net.init_pos_id(pos)
    net.plot_du_pos()


def plot_ref_event():
    pos, t_evt, val_evt = read_ref_data()
    plot_event(pos, t_evt, val_evt)


def use_plane_model():
    """
    Plan model
    """
    print("SPHERICAL PLANE")
    pos, t_evt, val_evt = read_ref_data()
    est_dir, chi2, res = pwf.solve_with_plane_model(pos, t_evt)
    print(np.rad2deg(est_dir))
    print("chi2=", chi2)


def use_spheric_model():
    """
    Spherical model
    """
    print("SPHERICAL MODEL")
    pos, t_evt, val_evt = read_ref_data()
    est_pars, chi2, _ = swf.solve_with_spheric_model(pos, t_evt)
    print(np.rad2deg(est_pars[:2]))
    print(est_pars)
    print("chi2=", chi2)


def use_spheric_model_class():
    """
    Spherical model
    """
    print("SPHERICAL MODEL")
    pos, t_evt, val_evt = read_ref_data()
    xmax_est = swf.ReconsXmaxSphericalModel(pos, t_evt)
    xmax_est.solve_xmax()
    print("Status solver: ", xmax_est.get_status_solver())
    print(xmax_est.rep_solver)
    print("Xmax: ", xmax_est.get_xmax())


def use_spheric_model_with_root_efield():
    """
    Spherical model
    """
    fef = FileSimuEfield(G_file_efield)
    tef = fef.get_obj_handling3dtraces()
    t_evt, _ = tef.get_tmax_vmax()
    pos = tef.network.du_pos
    # convert in s
    xmax_est = swf.ReconsXmaxSphericalModel(pos, t_evt / 1e9)
    xmax_est.plot_time_max()
    xmax_est.solve_xmax()
    xmax_est.net.plot_z()
    print(xmax_est.get_status_solver())
    print(xmax_est.rep_solver)
    print("Xmax: ", xmax_est.get_xmax())
    print("Xmax True: ", fef.tt_shower.xmax_pos_shc)

def use_plan_model_with_zhaires():
    """
    Spherical model
    """
    f_zh = ZhairesMaster(G_file_efield)
    i_sim = f_zh.get_simu_info()
    pprint.pprint(f_zh.get_simu_info())
    evt = f_zh.get_object_3dtraces()
    evt.plot_footprint_val_max() 
    t_evt, _ = evt.get_tmax_vmax()
    pos = evt.network.du_pos
    
    # convert in s
    #xmax_est = swf.ReconsXmaxSphericalModel(pos, t_evt / 1e9)
    est_dir, chi2, res = pwf.solve_with_plane_model(pos, t_evt)
    print(np.rad2deg(est_dir))
    print("chi2=", chi2)
    residu = pwf.pwf_residu(est_dir,pos, t_evt )
    evt.network.plot_footprint_1d(t_evt, "t_max",  scale="lin", unit="ns")
    evt.network.plot_footprint_1d(residu, "residu",  scale="lin", unit="ns")

if __name__ == "__main__":
    # read_ref_data()
    #plot_ref_event()
    use_plane_model()
    # use_spheric_model()
    # use_spheric_model_class()
    #use_spheric_model_with_root_efield()
    use_plan_model_with_zhaires()
    plt.show()
