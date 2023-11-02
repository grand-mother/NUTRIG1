'''
Created on 2 nov. 2023

@author: jcolley
'''

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from nutrig.flt.neural.cnn.ICRCNN_refact import remove_pic_near_border


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
    nb_bin = 32
    # Load nok data
    proba_ok = model.predict(data)
    hist_ok, bin_edges = np.histogram(proba_ok, nb_bin)
    dist_ok = hist_ok / hist_ok.sum()
    print('sum dist=', dist_ok.sum())
    return dist_ok, bin_edges


def get_separability(model, data_ok, data_nok):
    dist_ok, bin_edges = get_distrib(model, data_ok)
    dist_nok, bin_edges = get_distrib(model, data_nok)
    index_sep = np.sum(dist_ok * dist_nok)
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
    f_data_ok = datadir + 'day_simu_test_3.npy'
    f_model = datadir + 'trigger_icrc_80.keras'
    #
    model = load_model_ccn(f_model)
    data_ok = load_data(f_data_ok)
    data_nok = load_data(f_data_nok)
    get_separability(model, data_ok, data_nok)


if __name__ == '__main__':
    icrc_perfo()
    # 
    plt.show()
