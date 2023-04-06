"""

"""
import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data
from sradio.basis.du_network import DetectorUnitNetwork
import sradio.recons.shower_plane as pwf
import sradio.recons.shower_spheric as swf
from grand.io.root_files import FileSimuEfield


REF_POS = grand_add_path_data("tests/recons/ref_recons_coord_antennas.txt")
REF_EVENT = grand_add_path_data("tests/recons/ref_recons_coinctable.txt")

G_file_efield = "/home/dc1/Coarse2_xmax_add.root"
G_file_efield = "/home/dc1/Coarse3.root"


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
    est_dir, chi2 = pwf.solve_with_plane_model(pos, t_evt)
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


if __name__ == "__main__":
    # read_ref_data()
    plot_ref_event()
    # use_plane_model()
    # use_spheric_model()
    # use_spheric_model_class()
    use_spheric_model_with_root_efield()
    plt.show()
