'''
Created on 21 mai 2024

@author: jcolley
'''
import numpy as np
import matplotlib.pyplot as plt

from rshower.basis.traces_event import Handling3dTraces


def get_histo_snr(ds_bkg, ds_sig, name_ds="TBD", bins=[], sigma=10):
    assert isinstance(ds_bkg, Handling3dTraces)
    assert isinstance(ds_sig, Handling3dTraces)
    _, v_max_bkg = ds_bkg.get_tmax_vmax(False, "no")
    _, v_max_sig = ds_sig.get_tmax_vmax(False, "no")
    if bins==[]:
        bins = "auto"
    _, bin_edges = np.histogram(v_max_sig, bins)
    #hist_bkg, _ = np.histogram(v_max_bkg, bin_edges)
    plt.figure()
    plt.title(f"Histogram of dataset {name_ds} with SNR\n{ds_bkg.info_shower}\n{ds_sig.info_shower}")
    plt.title(f"Histogram of dataset {name_ds} with SNR")
    trans= 0.2
    plt.hist(v_max_sig/sigma, bin_edges, alpha=trans, label="Signal")
    plt.hist(v_max_bkg/sigma, bin_edges, alpha=trans, label="Background")
    plt.xlabel(f"SNR band")
    plt.ylabel("Number of trace in bin")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()