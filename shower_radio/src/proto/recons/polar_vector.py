"""
Created on 17 mars 2023

@author: jcolley
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from sradio.num.signal import filter_butter_band_lfilter
from sradio.num.signal import filter_butter_band_causal
from sradio.num.signal import filter_butter_band_causal_hc
from sradio.basis.traces_event import Handling3dTracesOfEvent
import sradio.io.sradio_asdf as fsr

from sradio.io.shower.zhaires_master import ZhairesMaster

a_inc = np.deg2rad(62.27)
a_inc = np.deg2rad(60.79)
V_B = np.array([np.cos(a_inc), 0, -np.sin(a_inc)])

f_asdf = "/home/jcolley/projet/nutrig_wk/NUTRIG1/shower_radio/src/proto/simu/out_v_oc.asdf"
#G_path_simu = "/home/jcolley/projet/grand_wk/bug/BugExample/Coarse2"
G_path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1"
G_path_simu = "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.296_74.8_0.0_1"

def load_file_trace(path_data=""):
    """
    return (n,3)

    :param path_data:
    """
    path_data = "/home/jcolley/projet/grand_wk/data/a2.trace"
    # path_data = "/home/jcolley/projet/grand_wk/data/zhaires/set500/GP300Outbox/GP300_Proton_3.97_74.8_0.0_1/GP300_Proton_3.97_74.8_0.0_1.traces/"
    path_data = (
        "/home/jcolley/projet/grand_wk/data/zhaires/Stshp_MZS_QGS204JET_Proton_0.21_56.7_90.0_5"
    )
    path_data = f"{path_data}/a24.trace"
    # path_data = f"{path_data}/a44.trace"
    t_trace = np.loadtxt(path_data)
    trace = t_trace[:, 1:]
    delta_ns = t_trace[1, 0] - t_trace[0, 0]
    f_sample_mhz = 1e-6 / (delta_ns * 1e-9)
    print(f_sample_mhz)
    return trace, f_sample_mhz


def plot_trace(t_3d, plt_tilte="", f_sample=1):
    h3dt = Handling3dTracesOfEvent(plt_tilte)
    s_tr = t_3d.shape[0]
    t_3d_a = np.zeros((1, 3, s_tr), dtype=np.float32)
    t_3d_a[0] = t_3d.transpose()
    print(t_3d_a.shape)
    h3dt.init_traces(t_3d_a, [0], np.array([0]), f_sample)
    h3dt.set_unit_axis(r"uV/m", "cart")
    h3dt.plot_trace_idx(0)
    if f_sample != 1:
        h3dt.plot_ps_trace_idx(0)


def test_fit_simu(plt_tilte):
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
    # print(f"loss : {s_residu}")
    return s_residu


def fit_linear_polar_fast_testB(efield_3d, threasold=20, v_b=V_B):
    """

    :param efield_3d: (n_s,3)
    """
    print("============fit_linear_polar FAST ===============")
    n_elec = np.linalg.norm(efield_3d, axis=1)
    idx_hb = np.where(n_elec > threasold)[0]
    p_raw = efield_3d[idx_hb].T / n_elec[idx_hb]
    a_raw_angle = np.rad2deg(np.arccos(np.dot(p_raw.T, v_b)))
    plt.figure()
    plt.title("angle(vec polar,B)")
    plt.hist(a_raw_angle)
    print(p_raw)
    idx_zn = np.where(p_raw[2] < 0)[0]
    print(idx_zn)
    # print(p_raw.T)
    p_raw = p_raw.T
    p_raw[idx_zn] = -p_raw[idx_zn]
    # print(p_raw.T)
    # weigth mean
    n_elec_s = n_elec[idx_hb] * n_elec[idx_hb]
    pol_est = np.sum(p_raw.T * n_elec_s, axis=1) / np.sum(n_elec_s)
    print(pol_est)
    p_vb = np.dot(pol_est, v_b)
    print(f"B.p = {p_vb}")
    a_pB = np.rad2deg(np.arccos(p_vb))
    print(f"angle(B,p)= {a_pB} deg")
    return pol_est

def estimate_polar_vec(trace, threasold=20):
    """

    :param efield_3d: (n_s,3)
    """
    print("===========estimate_polar_vec================")
    n_elec = np.linalg.norm(trace, axis=1)
    idx_hb = np.where(n_elec > threasold)[0]
    # unit
    sple_ok = trace[idx_hb].T / n_elec[idx_hb]
    idx_neg = np.where(sple_ok[1] < 0)[0]
    sple_ok = sple_ok.T
    sple_ok[idx_neg] = -sple_ok[idx_neg]
    n_elec_2 = n_elec[idx_hb] * n_elec[idx_hb]
    #
    pol_est = np.sum(sple_ok.T * n_elec_2, axis=1) / np.sum(n_elec_2)
    return pol_est

def fit_linear_polar(efield_3d, v_b=V_B):
    """

    :param efield_3d: (n,3)
    """
    print("============fit_linear_polar===============")
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
    print(res.message)
    residu = data[4]
    print(residu.shape)
    norm_p = np.linalg.norm(res.x)
    print(f"polar vec: p = {res.x}, norm={norm_p:.5f}")
    print(f"mag field: B = {v_b}")
    p_vb = np.dot(res.x / norm_p, v_b)
    print(f"B.p = {p_vb}")
    a_pB = np.rad2deg(np.arccos(p_vb))
    print(f"angle(B,p)= {a_pB} deg")
    if True:
        plt.figure()
        n_bin = 50
        plt.hist(residu[:, 0], n_bin, label="x", ls="dashed", lw=3, alpha=0.5, color="k")
        plt.hist(residu[:, 1], n_bin, label="y", ls="dashed", lw=3, alpha=0.5, color="y")
        plt.hist(residu[:, 2], n_bin, label="Z", ls="dashed", lw=3, alpha=0.5, color="b")
        plt.grid()
        plt.legend()

        plt.figure()
        plt.title("Trace in polarization frame")
        plt.plot(np.dot(efield_3d, -res.x), label="Polar E (3D => 1D)")
        plt.plot(n_elec, label="Norm E")
        plt.legend()
        plt.grid()


def test_polar_geo_mag(efield_3d, v_b=V_B):
    """

    :param efield_3d: (n_s,3)
    """
    print("=========test_polar_geo_mag==================")
    n_elec = np.linalg.norm(efield_3d, axis=1)
    print(efield_3d.shape, n_elec.shape)
    sum_n = np.sum(n_elec)
    print(sum_n)
    num_dot = np.sum(np.dot(efield_3d, v_b))
    b_e = num_dot / sum_n
    print(f"B.E_u = {b_e}")
    a_pB = np.rad2deg(np.arccos(b_e))
    print(f"angle(B,E)= {a_pB} deg")


def test_polar_geo_mag_cor(efield_3d, threasold=20, v_b=V_B):
    """

    :param efield_3d: (n_s,3)
    """
    print("=========test_polar_geo_mag CORRECTION==================")
    n_elec = np.linalg.norm(efield_3d, axis=1)
    idx_hb = np.where(n_elec > threasold)[0]
    # (3, ns)(ns)
    p_raw = (efield_3d[idx_hb].T / n_elec[idx_hb]).T
    a_cos = np.dot(p_raw, v_b)
    print(f"B.E_u = {a_cos.mean()}  +/-{a_cos.std()}")
    a_angle = np.rad2deg(np.arccos(a_cos))
    print(f"angle(B,E)= {a_angle.mean()} deg +/-{a_angle.std()}")
    test = a_angle.std() < 5
    test = test and np.fabs(a_angle.mean()-90) <5
    print(f"shower : {test}")
    plt.figure()
    plt.title("angle(B,E)")
    plt.hist(a_angle)


def test_raw_efield():
    trace, f_mhz = load_file_trace()
    plot_trace(trace, "raw", f_mhz)
    fit_linear_polar(trace)
    test_polar_geo_mag(trace)


def test_band_filter_efield():
    trace_raw, f_mhz = load_file_trace()
    # plot_trace(trace_raw, "raw", f_mhz)
    trace = filter_butter_band_causal(trace_raw, 50, 250, f_mhz, True)
    plot_trace(trace, "band filter", f_mhz)
    fit_linear_polar(trace)
    test_polar_geo_mag(trace)


def test_band_filter_efield_hc():
    trace_raw, f_mhz = load_file_trace()
    # plot_trace(trace_raw, "raw", f_mhz)
    trace = filter_butter_band_causal_hc(trace_raw, 50, 250, f_mhz, True)
    plot_trace(trace, "band filter hc", f_mhz)
    fit_linear_polar_fast(trace)
    fit_linear_polar(trace)
    test_polar_geo_mag(trace)
    test_polar_geo_mag_cor(trace)


def test_polar_efield(path_simu):
    f_zh = ZhairesMaster(path_simu)
    event = f_zh.get_object_3dtraces()
    d_simu = f_zh.get_simu_info()
    a_inc = d_simu["geo_mag2"]["inc"]
    print("inc B", a_inc)
    a_inc = np.deg2rad(a_inc)
    v_b = np.array([np.cos(a_inc), 0, -np.sin(a_inc)])
    print(a_inc, v_b)
    idx_du = 124
    event.plot_trace_idx(idx_du)
    trace = event.traces[idx_du].T
    fit_linear_polar_fast(trace, 50,  v_b=v_b)
    fit_linear_polar(trace, v_b)
    test_polar_geo_mag_cor(trace,50,  v_b=v_b)


def test_polar_voc(f_asdf):
    event, d_simu = fsr.load_asdf(f_asdf)
    assert isinstance(event, Handling3dTracesOfEvent)
    # event.plot_footprint_val_max()
    a_inc = d_simu["geo_mag2"]["inc"]
    print("inc B", a_inc)
    a_inc = np.deg2rad(a_inc)
    v_b = np.array([np.cos(a_inc), 0, -np.sin(a_inc)])
    print(a_inc, v_b)
    idx_du = 13
    event.plot_trace_idx(idx_du)
    trace = event.traces[idx_du].T
    fit_linear_polar_fast_testB(trace, 50, v_b=v_b)
    fit_linear_polar(trace, v_b)
    test_polar_geo_mag_cor(trace,50, v_b=v_b)


if __name__ == "__main__":
    np.random.seed(100)
    # test_raw_efield()
    # test_band_filter_efield()
    #test_band_filter_efield_hc()
    test_polar_efield(G_path_simu)
    #test_polar_voc(f_asdf)
    plt.show()
