"""

"""

import uproot
import awkward as ak

import time

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.efield_event import HandlingEfield
import rshower.io.events.rshower_asdf as rs_io

import nutrig.eval.cnn_ds_icrc as cnn_ds_icrc
import nutrig.eval.dataset as dataset

## GLOBAL

G_datadir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
G_f_model = G_datadir + "template_wpp_2l_150.keras"

# path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/sim_Xiaodushan_20221026_000000_RUN0_CD_ZHAireS-AN_0000/"
# path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAireS-AN/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0001/"
path_dc2 = "/home/jcolley/projet/grand_wk/data/root/dc2/ZHAireS/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0001/"
f_dc2_adc = "adc_5388-23832_L1_0000.root"
f_dc2_adc = "adc_39-24951_L1_0000.root"
f_dc2_ef = "efield_5388-23832_L0_0000.root"


quant = 2**13


def load_root_dc2_trace(pf_adc):
    # Read only traces
    fevt = uproot.open(pf_adc)
    ak_traces = fevt["tadc"]["trace_ch"].array()
    fevt.close()
    traces = ak.to_numpy(ak.flatten(ak_traces, 1))
    print("Read adc file : ", pf_adc)
    print(traces.dtype)
    print(traces.shape)
    return traces


def load_npz_dc2_trace(pf_adc):
    # Read only traces
    if pf_adc.find(".root"):
        pfile = pf_adc.replace(".root", ".npz")
    else:
        pfile = pf_adc
    data = np.load(pfile)["traces"]
    print(data.shape)
    return data


def load_asdf_dc2_trace(pf_adc):
    # Read only traces
    if pf_adc.find(".root"):
        pfile = pf_adc.replace(".root", ".asdf")
    else:
        pfile = pf_adc
    evt, _ = rs_io.load_asdf(pfile)
    print(evt.traces.shape)
    return evt


def get_distrib(model, data):
    nb_bin = 40
    # Load nok data
    t_cpu = time.process_time()
    proba_ok = model.predict(data)
    duration_cpu = time.process_time() - t_cpu
    print(f"Inference CPU time= {duration_cpu} s, for {data.shape[0]} traces")
    hist_ok, bin_edges = np.histogram(proba_ok, nb_bin)
    dist_ok = hist_ok / hist_ok.sum()
    print("sum dist=", dist_ok.sum())
    return dist_ok, bin_edges, proba_ok


def convert_numpy_file(pf_adc):
    traces = load_root_dc2_trace(pf_adc)
    assert traces.shape[2] == 2048
    traces = traces[:, :, :1024]
    pf_np = pf_adc.replace(".root", "")
    np.savez_compressed(pf_np, traces=traces)


def get_h3tr(pf_adc, name="", f_save=False):
    data = load_root_dc2_trace(pf_adc)
    h3tr = Handling3dTraces(name)
    assert data.shape[2] == 2048
    h3tr.init_traces(data[:, :, :1024], f_samp_mhz=500)
    h3tr.info_shower = pf_adc.split("/")[-1]
    h3tr.set_unit_axis("ADU", "dir", "ADC voltage")
    if f_save:
        rs_io.save_asdf_single_event(pf_adc.replace(".root", ""), h3tr, {})
    return h3tr


def plot_histo_proba(model, h3tr):
    assert isinstance(h3tr, Handling3dTraces)
    print(h3tr.get_nb_trace())
    # h3tr.remove_trace_low_signal(10 * 100)
    # m_norm = h3tr.get_max_norm()
    snr, noise, t_max, v_max = h3tr.get_snr_and_noise()
    print(np.min(snr))
    # h3tr.plot_trace_idx(0)
    # h3tr.plot_trace_idx(11)
    # h3tr.plot_trace_idx(101)
    # h3tr.plot_trace_idx(201)
    # h3tr.plot_trace_idx(301)
    snr = np.squeeze(snr)
    data_dc2 = np.swapaxes(h3tr.traces, 1, 2) / quant
    dist_all, bin_edges, proba_all = get_distrib(model, data_dc2)
    proba_all = np.squeeze(proba_all)
    nb_trace = h3tr.get_nb_trace()
    plt.figure()
    plt.title(f"CNN inference, proba is shower\n{nb_trace} traces")
    plt.hist(proba_all)
    plt.yscale("log")
    plt.grid()
    plt.figure()
    plt.title(f"Data set, max value trace\n{nb_trace} traces")
    plt.hist(v_max)
    plt.yscale("log")
    plt.grid()
    plt.figure()
    plt.title("DC2 ADC, SNR trace histogram")
    plt.hist(snr, 40)
    plt.yscale("log")
    plt.xlabel("SNR")
    plt.grid()
    plt.figure()
    plt.title("DC2 , SNR versus CNN proba")
    plt.plot(snr, proba_all, "*")
    plt.xscale("log")
    plt.xlabel("SNR")
    plt.ylabel("CNN probability")
    plt.vlines(16, 0.1, 1, label="SNR 10, after only True pulse",colors='y', linestyles="-.")
    plt.grid()
    plt.legend()

    l_band = [0, 6, 8, 16, 32, 64, 128, 2000]
    l_band = [0, 6,8,9, 10]
    # l_band = [0, 3, 6, 10, 20, 40]
    # idx_ok = h3tr.remove_trace_low_signal(3, snr)
    # v_max = v_max[idx_ok]
    for idx, edge in enumerate(l_band):
        if idx == 0:
            continue
        s_band = f"SNR [{l_band[idx-1]},{l_band[idx]}]"
        print(f"Processing {s_band}")
        print(snr.shape)
        idx_ok = np.nonzero(np.logical_and(snr > l_band[idx - 1], snr < l_band[idx]))[0]
        print(idx_ok.shape)
        proba_ok = proba_all[idx_ok]
        plt.figure()
        plt.title(f"Time of max versus CNN probability, {s_band}")
        t_tot = h3tr.get_size_trace() * h3tr.get_delta_t_ns()[0]
        plt.plot(100 * (t_max[idx_ok] - h3tr.t_start_ns[idx_ok]) / t_tot, proba_ok, "*")
        plt.grid()
        plt.xlim([0, 100])
        plt.ylim([0, 1.1])
        vline = [40.5, "Around time of max in simulation"]
        plt.vlines(vline[0], 0, 100, label=vline[1], linestyles="-.")
        plt.xlabel("Position of time of max in trace, % ")
        plt.ylabel(f"CNN probability")
        tr_band = Handling3dTraces(f"{s_band}")
        tr_band.init_traces(h3tr.traces[idx_ok], f_samp_mhz=h3tr.f_samp_mhz[0])
        tr_band.get_tmax_vmax(False, "no")
        hline = [3 * noise[idx_ok].mean(), "Noise 3$\sigma$"]
        # tr_band.plot_tmax_vmax(hline, vline)
        # histo proba in band
        plt.figure()
        nb_trace = proba_ok.shape[0]
        plt.title(f"CNN probability in {s_band}\n{nb_trace} traces")
        plt.hist(proba_ok, 30, range=[0, 1.01])
        plt.xlim([0, 1.1])
        plt.yscale("log")
        plt.grid()
        print(s_band, proba_ok)
        # some traces
        idx = idx_ok[0]
        h3tr.name = f"{s_band}, CNN proba {proba_all[idx]:.3f}"
        h3tr.plot_trace_idx(idx)
        idx = idx_ok[-1]
        h3tr.name = f"{s_band}, CNN proba {proba_all[idx]:.3f}"
        # h3tr.plot_trace_idx(idx)


if __name__ == "__main__":
    f_model = G_datadir + "template_wpp_2l_150.keras"
    model = keras.models.load_model(f_model)
    fadc = path_dc2 + f_dc2_adc
    # convert_numpy_file(fadc)
    # data = load_npz_dc2_trace(fadc)
    # h3tr = get_h3tr(fadc, f_save=False)
    h3tr = load_asdf_dc2_trace(fadc)
    plot_histo_proba(model, h3tr)
    #
    plt.show()
