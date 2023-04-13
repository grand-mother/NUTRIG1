"""
Created on 11 avr. 2023

@author: jcolley
"""


import numpy as np
import matplotlib.pylab as plt

from sradio.io.shower.zhaires_master import ZhairesMaster
import sradio.manage_log as mlg


CCIN2P3 = False


L_path_data = []


if CCIN2P3:
    path_data = ""
else:
    L_path_data.append(
        "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
    )
    L_path_data.append(
        "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
    )
    path_data = L_path_data[0]

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True, log_root="proto")


def select_simu_azimuth():
    pass

def load_event(path_event):
    zhaires = ZhairesMaster(path_event)
    # tr_evt.trace is (n_du, 3, n_s)
    evt = zhaires.get_object_3dtraces()
    d_simu = zhaires.get_simu_info()
    return evt, d_simu


def check_polar_one_event(evt, d_simu):
    logger.info(evt.du_id[:5])
    # B definition
    a_inc = d_simu["geo_mag2"]["inc"]
    logger.info(f"inclinaison mag = {a_inc} deg")
    a_inc = np.deg2rad(a_inc)
    v_b = np.array([np.cos(a_inc), 0, -np.sin(a_inc)])
    # norm_tr (n_du, n_s)
    norm_tr = evt.get_norm()
    logger.info(norm_tr.shape)
    sum_n = np.sum(norm_tr, axis=1)
    logger.info(sum_n[:5])
    # traces (n_du,3,n_s) and v_b (3,) => need to swap for dot function
    new_traces = np.swapaxes(evt.traces, 1, 2)
    num_dot = np.sum(np.dot(new_traces, v_b), axis=1)
    logger.info(num_dot.shape)
    b_e = num_dot / sum_n
    logger.info(f"B.E_u = {b_e[:5]}")
    a_pB = np.rad2deg(np.arccos(b_e))
    logger.info(a_pB[:5])
    return d_simu["shower_azimuth"], a_pB


def plot_box_single_angle(azi, a_pB):
    plt.figure()
    plt.title("Angle (B,E) estimation in ZHAireS simulation ")
    plt.boxplot(a_pB, whis=[1, 99], positions=[azi], autorange=True, showmeans=True)
    plt.xlabel("Azimuth, degree")
    plt.ylabel("degree, (box plot 1%, 99%)")
    plt.grid()


def plot_box_angles_be(l_azi, l_abe):
    plt.figure()
    plt.title("Angle (B,E) estimation in ZHAireS simulation ")
    plt.xlim([-5, 100])
    plt.boxplot(l_abe, whis=[1, 99], widths=7, positions=l_azi, autorange=True, showmeans=True)
    plt.xlabel("Azimuth, degree")
    plt.ylabel("degree, (box plot 1%, 99%)")
    plt.grid()


def master_check_multi_events(l_path_data):
    l_azi = []
    l_angle_be = []
    for idx, path_data in enumerate(l_path_data):
        evt, d_simu = load_event(path_data)
        azi, angles = check_polar_one_event(evt, d_simu)
        l_azi.append(azi)
        l_angle_be.append(angles)
    plot_box_angles_be(l_azi, l_angle_be)


def master_check_one_event(path_data):
    evt, d_simu = load_event(path_data)
    azi, angles = check_polar_one_event(evt, d_simu)
    plot_box_single_angle(azi, angles)


if __name__ == "__main__":
    # master_check_one_event(path_data)
    master_check_multi_events(L_path_data)
    plt.show()
