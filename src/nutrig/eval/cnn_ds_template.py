"""
Created on 2 nov. 2023

@author: jcolley
"""

import time

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from rshower.basis.traces_event import Handling3dTraces

import nutrig.eval.cnn_ds_icrc as cnn_ds_icrc
import nutrig.eval.dataset as dataset

## GLOBAL

G_datadir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
quant = 2**13

f_model = G_datadir + "template_wpp_2l_150.keras"
f_test_ok = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
f_test_nok = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
f_train_ok = G_datadir + "sig_dataset_nutrig_gp13_train_seed_300.npz"
f_train_nok = G_datadir + "bkg_dataset_nutrig_gp13_train_seed_300.npz"


def get_data_template(f_data, normalize=True):
    data_cnn = np.load(f_data)["traces"]
    data_cnn = np.swapaxes(data_cnn, 1, 2)
    if normalize:
        data_cnn = data_cnn / quant
    return data_cnn

def get_h3tr_template(f_data, name=""):
    data = np.load(f_data)["traces"]
    h3tr = Handling3dTraces(name)
    h3tr.init_traces(data, f_samp_mhz=500)
    h3tr.info_shower = f_data.split("/")[-1]
    h3tr.set_unit_axis("ADU", "cart", "ADC voltage")
    return h3tr
    
def write_prob_file(f_model, f_data, f_proba):
    model = keras.models.load_model(f_model)
    data_cnn = np.load(f_data)["traces"]
    data_cnn = np.swapaxes(data_cnn, 1, 2)
    data_cnn = data_cnn / quant
    proba = model.predict(data_cnn)
    np.save(f_proba, proba)
    # plot distrib
    hist_ok, bin_edges = np.histogram(proba, 40)
    dist_ok = hist_ok / hist_ok.sum()
    print("sum dist=", dist_ok.sum())
    plt.figure()
    plt.title(f"Distribution {f_data.split('/')[-1]}")
    plt.semilogy(bin_edges[1:], dist_ok)
    plt.grid()
    plt.legend()


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


def get_proba_template_sig():
    f_model = G_datadir + "template_wpp_2l_150.keras"
    f_data = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
    f_proba = "sig_proba_cnn_nutrig_gp13_test_seed_300"
    write_prob_file(f_model, f_data, f_proba)


def get_proba_template_bkg():
    f_model = G_datadir + "template_wpp_2l_150.keras"
    f_data = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
    f_data = G_datadir + "bkg_dataset_nutrig_gp13_th1_55_th2_35_test_seed_300.npz"
    f_proba = "bkg_proba_cnn_nutrig_gp13_test_seed_300"
    f_proba = "bkg_proba_cnn_nutrig_gp13_th1_55_th2_35_test_seed_300"
    write_prob_file(f_model, f_data, f_proba)


def get_sepabability_template():
    f_model = G_datadir + "template_wpp_2l_150.keras"
    f_data_ok = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
    f_data_nok = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
    data_ok = get_data_template(f_data_ok)
    data_nok = get_data_template(f_data_nok)
    model = keras.models.load_model(f_model)
    cnn_ds_icrc.get_separability(model, data_ok, data_nok, f_data_ok.split("/")[-1])

def plot_dataset_snr():
    bins = [1,4,5,6,7,8,9,10,13,17,20,30,40,50,100]
    hsig = get_h3tr_template(f_test_ok, "Signal test")
    hbkg = get_h3tr_template(f_test_nok, "Background test")    
    dataset.get_histo_snr(hbkg, hsig, "Test Template", bins, sigma=10)
    hsig = get_h3tr_template(f_train_ok, "Signal training")
    hbkg = get_h3tr_template(f_train_nok, "Background training")
    dataset.get_histo_snr(hbkg, hsig, "Training Template", bins, sigma=10)
    #dataset.get_histo_snr(hbkg, hsig, "Test Template2", sigma=10)

def get_sepabability_snr_template():
    '''
    quick and dirty separability index versus SNR
    '''
    f_model = G_datadir + "template_wpp_2l_150.keras"
    f_data_ok = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
    f_data_nok = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
    model = keras.models.load_model(f_model)
    #
    l_snr = [4,5,6,7,9,10]
    #l_snr = [4,10]
    l_is = []
    l_nb_trace = []
    sigma_noise = 10
    for snr in l_snr:
        threshold = snr*sigma_noise
        data_ok = get_data_template(f_data_ok, False)
        trok = Handling3dTraces("Signal")
        trok.init_traces(np.swapaxes(data_ok, 1, 2), f_samp_mhz=500)
        trok.get_tmax_vmax(False, "no")
        idx = 1201
        # trok.plot_psd_trace_idx(idx)
        # trok.plot_all_traces_as_image()
        trok.remove_trace_low_signal(threshold)
        nb_ok = trok.get_nb_trace()
        trok.get_tmax_vmax(False, "no")
        trok.plot_trace_idx(snr)
        #trok.plot_all_traces_as_image()
        data_ok = np.swapaxes(trok.traces, 1, 2)/quant
        #
        data_nok = get_data_template(f_data_nok, False)
        trnok = Handling3dTraces("Background only")
        trnok.init_traces(np.swapaxes(data_nok, 1, 2), f_samp_mhz=500)
        trnok.remove_trace_low_signal(threshold)
        nb_nok = trnok.get_nb_trace()
        #trnok.plot_all_traces_as_image()
        data_nok = np.swapaxes(trnok.traces, 1, 2)/quant        
        res = cnn_ds_icrc.get_separability(model, data_ok, data_nok, f_data_ok.split("/")[-1], False)
        l_is.append(res[0])
        l_nb_trace.append([nb_ok, nb_nok])
    
    plt.figure()
    plt.title("Separability index for dataset 'Template' versus SNR")
    plt.plot(l_snr, l_is)
    plt.grid()
    plt.ylabel("Separability index\n(1 perfect)")
    plt.xlabel("SNR threshold\n(sigma noise 10 ADU, signal max ||trace x,y,z||)")
    a_nb = np.array(l_nb_trace)
    plt.figure()
    plt.title("Number of traces in dataset test 'Template' versus SNR")
    plt.plot(l_snr, a_nb[:,0], label="Signal")
    plt.plot(l_snr, a_nb[:,1], label="Background")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.ylabel("Number of traces")
    plt.xlabel("SNR threshold\n(sigma noise 10 ADU, signal max ||trace x,y,z||)")

if __name__ == "__main__":

    # get_proba_template_sig()
    # get_sepabability_template()
    get_sepabability_snr_template()
    #plot_dataset_snr()
    #
    plt.show()
