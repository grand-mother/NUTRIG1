"""

"""

import awkward as ak
import uproot

import time

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from rshower.basis.traces_event import Handling3dTraces
from rshower.basis.efield_event import HandlingEfield
import rshower.io.events.rshower_asdf as rs_io
from rshower.io.events.grand_trigged import GrandEventsSelectedFmt01

import nutrig.eval.cnn_ds_icrc as cnn_ds_icrc
import nutrig.eval.dataset as dataset
from nutrig.eval.common import get_distrib, quant

## GLOBAL

G_datadir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
G_f_model = G_datadir + "template_wpp_2l_150.keras"

pn_fevents = (
    "/home/jcolley/projet/grand_wk/data/event/gp13_2024_polar/GP13_UD_240616_240708_with_time.npz"
)


def plot_cnn_proba(model, pn_fevents):
    # snr, noise, t_max, v_max = h3tr.get_snr_and_noise()
    # print(np.min(snr))
    # plt.title("GP13 ADC, SNR trace histogram")
    # plt.hist(snr, 40)
    # plt.yscale("log")
    # plt.xlabel("SNR")
    # plt.grid()
    #
    # plt.figure()
    # plt.title("GP13 events, SNR versus CNN proba")
    # plt.plot(snr, proba_all, "*")
    # plt.xscale("log")
    # plt.xlabel("SNR")
    # plt.ylabel("CNN probability")
    # plt.vlines(16, 0.1, 1, label="SNR 10, after only True pulse",colors='y', linestyles="-.")
    # plt.grid()
    # plt.legend()
        
    # Load data to init Handling3dTraces object
    df_events = GrandEventsSelectedFmt01(pn_fevents)
    plt.figure()
    cpt_tot = 0
    cpt_ok = 0
    for i_e in range(df_events.nb_events):
        evt = df_events.get_3dtraces(i_e)
        #evt.plot_trace_idx(2)
        data = np.swapaxes(evt.traces, 1, 2) / quant
        dist_ok, bin_edges, proba_ok = get_distrib(model, data)
        print(f"{i_e} : {proba_ok}")
        for i_d in range(evt.get_nb_trace()):
            cpt_tot += 1
            plt.plot(i_e, proba_ok[i_d], "*")
            if proba_ok[i_d] > 0.6:
                cpt_ok += 1 
    plt.grid()
    plt.title(f"CNN on GP13 events, {cpt_ok} ok on {cpt_tot} traces")
    plt.ylim([-.05, 1.05])
    plt.ylabel("CNN probability")
    plt.xlabel("Index event")
    #plt.legend()
    l_proba_snr = []
    for i_e in range(df_events.nb_events):
        evt = df_events.get_3dtraces(i_e)
        snr, noise_mean, t_max, v_max = evt.get_snr_and_noise()
        #evt.plot_trace_idx(2)
        data = np.swapaxes(evt.traces, 1, 2) / quant
        dist_ok, bin_edges, proba_ok = get_distrib(model, data)
        l_proba_snr.append([snr, proba_ok])
        print(f"{i_e} : {proba_ok}")
        for i_d in range(evt.get_nb_trace()):
            if proba_ok[i_d] < 0.2:
                evt.plot_trace_idx(i_d)
            if snr[i_d] > 40:
                evt.plot_trace_idx(i_d)
    plt.figure()
    plt.title("")
    for i_e in range(df_events.nb_events):
        plt.plot(l_proba_snr[i_e][0], l_proba_snr[i_e][1],'*')
        
    plt.xlabel("SNR")
    plt.ylabel("CNN probability")
    plt.grid()
    
    


if __name__ == "__main__":
    f_model = G_datadir + "template_wpp_2l_150.keras"
    model = keras.models.load_model(f_model)
    plot_cnn_proba(model, pn_fevents)
    #
    plt.show()
