"""
Created on 17 mars 2023

@author: jcolley
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.spatial.transform import Rotation


def load_file_trace(path_data=""):
    path_data = "/home/jcolley/projet/grand_wk/data/a2.trace"
    path_data = "/home/jcolley/projet/grand_wk/data/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1/GP300_Proton_3.97_74.8_0.0_1.traces/"
    path_data = f"{path_data}/a80.trace"
    t_trace = np.loadtxt(path_data)
    trace = t_trace[:, 1:]
    return trace


def plot_trace(t_3d, plt_tilte=""):
    plt.figure()
    plt.title(plt_tilte)
    plt.plot(t_3d[:, 0], label="col 1")
    plt.plot(t_3d[:, 1], label="col 2")
    plt.plot(t_3d[:, 2], label="col 3")
    plt.legend()
    plt.grid()


def test_fit_simu():
    nb_sample = 500
    a_time_ns = np.arange(nb_sample)
    max_ef = 100
    c_light = 300000.0
    c_light_m_ns = c_light / 1e6
    w_length_m = 10
    ef_x = max_ef * np.cos(2 * np.pi * c_light_m_ns / w_length_m * a_time_ns)
    a_zeros = np.zeros(nb_sample)
    ef = np.array([ef_x, a_zeros, a_zeros]).transpose()
    # print(ef.shape)
    plot_trace(ef)
    ef += np.random.normal(0, 10, (nb_sample, 3))
    plot_trace(ef)
    return ef


def test_fit_simu_rot():
    ef = test_fit_simu()
    mr2 = Rotation.from_euler("zy", [70, -70], degrees=True)
    v_x = np.array([1, 0, 0])
    print(mr2.as_matrix() @ v_x)
    rot_ef = mr2.as_matrix() @ ef.T
    plot_trace(rot_ef.T)
    return rot_ef.T


def loss_function_w_vect(data, w_vect):
    """
    data   : [ efield (n,3) ,  norm (n,) , sum norm , nb sample]
    w_vect : wave vector (3,) unit vector
    """
    elec = data[0]
    n_elec = data[1]
    sum_n = data[2]
    res = np.dot(elec, w_vect)
    dot_p = res * n_elec
    residu = np.sum(dot_p * dot_p)
    constraint = 1 - (w_vect * w_vect).sum()
    weight = 1000
    loss = residu + weight * constraint
    print(f"loss : {loss}, {residu} {constraint}")
    return loss


def loss_function_lin_pol(v_pol, data):
    """
    data   : [ efield (n,3) ,  norm (n,) , sum norm , nb sample]
    w_vect : wave vector (3,) unit vector
    """
    elec = data[0]
    n_elec = data[1]
    sum_n = data[2]
    # print(elec[:4])
    coef = np.dot(elec, v_pol)
    # print(res.shape, v_pol.shape)
    # print(res[:4], v_pol)
    res = np.outer(coef, v_pol) - elec
    data[4] = res
    s_residu = np.sum(res * res) / data[3]
    # constraint = 1 - np.sqrt((v_pol * v_pol).sum())
    # constraint = 100000 * constraint ** 2
    # print(f'loss : {s_residu}, {constraint} {v_pol[2]}')
    print(f"loss : {s_residu}")
    return s_residu


def fit_linear_polar(efield_3d):
    plot_trace(efield_3d)
    n_elec = np.linalg.norm(efield_3d, axis=1)
    sum_n = np.sum(n_elec)
    data = [efield_3d, n_elec, sum_n, n_elec.shape[0], 0]
    # wave vector guess
    guess = np.array([0, 0, 1])
    # bounds = Bounds([-1, -1, -1], [1, 1, 1])
    loss_function_lin_pol(guess, data)
    # res = minimize(loss_function_lin_pol, guess, method='trust-constr',
    #                args=data,bounds=bounds,
    #                options={'disp': True, 'xtol': 1e-6})
    res = minimize(loss_function_lin_pol, guess, method="BFGS", args=data, options={"disp": True})
    print(res.x, np.linalg.norm(res.x))
    print(res.message)
    residu = data[4]
    print(residu.shape)
    plt.figure()
    n_bin = 50
    plt.hist(residu[:, 0], n_bin)
    plt.figure()
    plt.hist(residu[:, 1], n_bin)
    plt.figure()
    plt.hist(residu[:, 2], n_bin)


if __name__ == "__main__":
    np.random.seed(100)
    # ef = test_fit_simu()
    # fit_wave_vector(ef)
    # r_ef = test_fit_simu_rot()
    # fit_linear_polar(r_ef)
    trace = load_file_trace()
    fit_linear_polar(trace)
    plt.show()
