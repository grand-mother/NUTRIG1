'''
Created on 2 nov. 2023

@author: jcolley
'''

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from nutrig.flt.neural.cnn.ICRCNN_refact import remove_pic_near_border
from sradio.basis.traces_event import Handling3dTracesOfEvent



def load_model_ccn(f_model):
    model = keras.models.load_model(f_model)
    return model


def load_data(f_data):
    data_cnn = np.load(f_data)
    data_cnn = remove_pic_near_border(data_cnn)
    quant = 2 ** 13
    data_cnn = data_cnn / quant
    return data_cnn


def get_distrib(model, data):
    nb_bin = 50
    # Load nok data
    proba_ok = model.predict(data)
    hist_ok, bin_edges = np.histogram(proba_ok, nb_bin)
    dist_ok = hist_ok / hist_ok.sum()
    print('sum dist=', dist_ok.sum())
    return dist_ok, bin_edges, proba_ok


def plot_shower_trigged(data, proba_shower):
    event = Handling3dTracesOfEvent("trigger OK")
    event_nok = Handling3dTracesOfEvent("trigger **NOK**")
    print(proba_shower.shape)
    print(data.shape)
    max_proba = np.max(proba_shower)
    min_proba = np.min(proba_shower)
    data_shower =  data[(proba_shower == max_proba)]
    data_noshower =  data[(proba_shower < 0.1)]
    print(data_noshower.shape)
    event.init_traces(np.swapaxes(data_shower,1,2))
    event_nok.init_traces(np.swapaxes(data_noshower,1,2))
    for idx in range(event.get_nb_du()):
        #event.plot_trace_idx(idx)
        pass
    for idx in range(20):
        event_nok.plot_trace_idx(idx)


def get_separability(model, data_ok, data_nok):
    dist_ok, bin_edges, _ = get_distrib(model, data_ok)
    dist_nok, bin_edges, _ = get_distrib(model, data_nok)
    index_sep = 1- np.sqrt(np.sum(dist_ok * dist_nok))
    print('index_sep=', index_sep)
    plt.figure()
    plt.title("Distribution")
    plt.semilogy(bin_edges[1:], dist_ok, label="shower")
    plt.semilogy(bin_edges[1:], dist_nok, label="background")
    plt.grid()
    plt.legend()
    plt.figure()
    plt.title("Distribution")
    plt.plot(bin_edges[1:], dist_ok, label="shower")
    plt.plot(bin_edges[1:], dist_nok, label="background")
    plt.grid()
    plt.legend()
    return index_sep, dist_ok, dist_nok, bin_edges

    
def icrc_perfo():
    datadir = '/home/jcolley/projet/grand_wk/data/npy/'
    f_data_nok = datadir + 'day_backg_test.npy'
    f_data_ok = datadir + 'day_simu_test_8+.npy'
    #f_data_ok = datadir + 'day_simu_test_3.npy'
    f_model = datadir + 'trigger_icrc_80.keras'
    #
    model = load_model_ccn(f_model)
    data_ok = load_data(f_data_ok)
    data_nok = load_data(f_data_nok)
    # 
    dist_nok, bin_edges, proba_nok = get_distrib(model, data_nok)
    plot_shower_trigged(data_nok, proba_nok[:,0])


if __name__ == '__main__':
    icrc_perfo()
    # 
    plt.show()
