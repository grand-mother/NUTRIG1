'''
Created on 2 nov. 2023

@author: jcolley
'''

from tensorflow import keras
import numpy as np


def load_model_ccn(f_model):
    model = keras.models.load_model(f_model)

def load_data(f_data):
    data_cnn = None
    return data_cnn


def get_separability(model, data_ok, data_nok):
    nb_bin  = 32
    # Load nok data
    proba_ok = model.predict(data_ok)
    hist_ok, bin_edges= np.histogram(proba_ok, nb_bin)
    dist_ok = hist_ok()/hist_ok.sum()
    # plot dist NOK
    # load ok data
    dist_ok = model.predict(data_nok)
    index_sep = np.sum(dist_ok*dist_nok)
    return index_sep
    

def predict_model(model):
    pass

if __name__ == '__main__':
    pass