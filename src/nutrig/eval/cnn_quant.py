'''
Created on 2 nov. 2023

@author: jcolley
'''
import time


from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from nutrig.flt.neural.cnn.ICRCNN_refact import remove_pic_near_border, save_trace
from sradio.basis.traces_event import Handling3dTracesOfEvent


G_datadir = '/home/jcolley/projet/grand_wk/data/npy/' 

def tflite_inference_fp16_REF(input_data, f_tf="converted_model.tflite"):
    """
    input_data shape (1024,3) only
    """
    print(input_data.dtype)
    print(input_data.shape)
    # Load the TFLite model and allocate tensors.
    interpreter_fp16 = tf.lite.Interpreter(model_path=f_tf)
    interpreter_fp16.allocate_tensors()
    output_index = interpreter_fp16.get_output_details()[0]["index"]
    input_index = interpreter_fp16.get_input_details()[0]["index"]
    print(input_data.shape)
    print(input_data[0].shape)
    input_fmt = np.expand_dims(input_data[0], axis=0).astype(np.float32)
    print(input_fmt.shape)
    interpreter_fp16.set_tensor(input_index, input_fmt)
    interpreter_fp16.invoke()
    output = interpreter_fp16.get_tensor(output_index)
    print(output.shape)
    print(output.dtype)
    print(output)

def tflite_inference_fp16(input_data, f_tf="converted_model.tflite"):
    print(input_data.dtype)
    print(input_data.shape)
    # Load the TFLite model and allocate tensors.
    interpreter_fp16 = tf.lite.Interpreter(model_path=f_tf)
    interpreter_fp16.allocate_tensors()
    output_index = interpreter_fp16.get_output_details()[0]["index"]
    input_index = interpreter_fp16.get_input_details()[0]["index"]
    input_fmt = np.expand_dims(input_data[0], axis=0).astype(np.float32)
    interpreter_fp16.set_tensor(input_index, input_fmt)
    interpreter_fp16.invoke()
    output = interpreter_fp16.get_tensor(output_index)
    return output

def load_model_cnn(f_model):
    model = keras.models.load_model(f_model)
    return model


def load_data_and_preproc(f_data):
    data_cnn = np.load(f_data)
    data_cnn = remove_pic_near_border(data_cnn)
    quant = 2 ** 13
    data_cnn = data_cnn / quant
    return data_cnn


def get_distrib(model, data):
    nb_bin = 42
    # Load nok data
    t_cpu = time.process_time() 
    proba_ok = model.predict(data)
    duration_cpu = time.process_time() - t_cpu
    print(f"CPU time= {duration_cpu} s")
    hist_ok, bin_edges = np.histogram(proba_ok, nb_bin)
    dist_ok = hist_ok / hist_ok.sum()
    print('sum dist=', dist_ok.sum())
    return dist_ok, bin_edges, proba_ok


def concatenate_test_ok():
    pf_data_ok = G_datadir + 'day_simu_test_3.npy'
    data_ok = load_data_and_preproc(pf_data_ok)
    print(data_ok.shape)
    pf_data_ok = G_datadir + 'day_simu_test_4.npy'
    temp = load_data_and_preproc(pf_data_ok)
    print(temp.shape)
    data_ok = np.concatenate((data_ok, temp), axis=0)
    pf_data_ok = G_datadir + 'day_simu_test_5.npy'
    temp = load_data_and_preproc(pf_data_ok)
    print(temp.shape)
    data_ok = np.concatenate((data_ok, temp), axis=0)
    pf_data_ok = G_datadir + 'day_simu_test_6.npy'
    temp = load_data_and_preproc(pf_data_ok)
    print(temp.shape)
    data_ok = np.concatenate((data_ok, temp), axis=0)
    pf_data_ok = G_datadir + 'day_simu_test_7.npy'
    temp = load_data_and_preproc(pf_data_ok)
    print(temp.shape)
    data_ok = np.concatenate((data_ok, temp), axis=0)
    pf_data_ok = G_datadir + 'day_simu_test_8+.npy'
    temp = load_data_and_preproc(pf_data_ok)
    print(temp.shape)
    data_ok = np.concatenate((data_ok, temp), axis=0)
    print(data_ok.shape)
    return data_ok


def plot_shower_trigged(data, proba_shower):
    event = Handling3dTracesOfEvent("trigger OK")
    event_nok = Handling3dTracesOfEvent("trigger **NOK**")
    print(proba_shower.shape)
    print(data.shape)
    max_proba = np.max(proba_shower)
    min_proba = np.min(proba_shower)
    data_shower = data[(proba_shower == max_proba)]
    data_noshower = data[(proba_shower < 0.2)]
    print(data_noshower.shape)
    event.init_traces(np.swapaxes(data_shower, 1, 2))
    event_nok.init_traces(np.swapaxes(data_noshower, 1, 2))
    for idx in range(10):
        event.plot_trace_idx(idx)
        pass
    for idx in range(10):
        event_nok.plot_trace_idx(idx)
        pass


def aaget_separability(model, data_ok, data_nok, f_data_ok=""):
    dist_ok, bin_edges, _ = get_distrib(model, data_ok)
    dist_nok, bin_edges, _ = get_distrib(model, data_nok)
    index_sep = 1 - np.sqrt(np.sum(dist_ok * dist_nok))
    #index_sep = 1 - np.sum(dist_ok * dist_nok)
    print("nb dist_ok: ", dist_ok.shape[0])
    print("nb dist_Nok: ", dist_nok.shape[0])
    print('index_sep=', index_sep)
    raise
    plt.figure()
    plt.title("Distribution")
    plt.semilogy(bin_edges[1:], dist_ok, label=f"shower {f_data_ok}")
    plt.semilogy(bin_edges[1:], dist_nok, label="background")
    plt.grid()
    plt.legend()
    plt.figure()
    plt.title("Distribution")
    plt.plot(bin_edges[1:], dist_ok, label=f"shower {f_data_ok}")
    plt.plot(bin_edges[1:], dist_nok, label="background")
    plt.grid()
    plt.legend()
    return index_sep, dist_ok, dist_nok, bin_edges


def aaicrc_perfo_all(file_model):
    pf_data_nok = G_datadir + 'day_backg_test.npy'
    f_model = G_datadir + file_model
    #f_model = G_datadir + 'trigger_icrc_80_acc96.keras'
    #
    model = load_model_cnn(f_model)
    data_ok = concatenate_test_ok()
    data_nok = load_data_and_preproc(pf_data_nok)
    get_separability(model, data_ok, data_nok)
    return
    dist_nok, bin_edges, proba_nok = get_distrib(model, data_nok)
    idx=10
    print(proba_nok.shape)
    pba = proba_nok[idx,0]
    save_trace(f"trace_{pba:.6f}", data_nok, idx)
    print(proba_nok[idx])
    # 
    dist_nok, bin_edges, proba_nok = get_distrib(model, data_nok)
    dist_ok, bin_edges, proba_ok = get_distrib(model, data_ok)
    print(bin_edges.shape, dist_nok.shape)
    bin_center = (bin_edges[:-1]+bin_edges[1:])/2
    print(bin_center.shape, bin_center)
    nb_bin = bin_center.shape[0]
    eff_txevt = np.empty((nb_bin, 2), dtype=np.float32)
    for idx in range(nb_bin):
        eff_txevt[idx,0] = np.sum(dist_ok[idx:])
        eff_txevt[idx,1] = 1000*np.sum(dist_nok[idx:])
    return eff_txevt
            
            
        # 
        # plot_shower_trigged(data_ok, proba_ok[:, 0])
def test_quant():
    pf_data_nok = G_datadir + 'day_backg_test.npy'
    f_model = G_datadir + 'test2_fmt.tflite'
    #f_model = G_datadir + 'trigger_icrc_80_acc96.keras'
    #
    #model = load_model_cnn(f_model)
    data_ok = concatenate_test_ok()
    data_nok = load_data_and_preproc(pf_data_nok)
    tflite_inference_fp16(data_ok[:10],f_model )
    #get_separability(model, data_ok, data_nok)


def icrc_perfo():
    
    pf_data_nok = G_datadir + 'day_backg_test.npy'
    f_data_ok = 'day_simu_test_8+.npy'
    #f_data_ok = 'day_simu_test_3.npy'
    pf_data_ok = G_datadir + f_data_ok
    #f_model = G_datadir + "trigger_pulse256_512_120.keras"
    f_model = G_datadir + 'trigger2l.keras'
    #f_model = G_datadir + 'trigger_icrc_80_acc96.keras'
    #
    model = load_model_cnn(f_model)
    data_ok = load_data_and_preproc(pf_data_ok)
    data_nok = load_data_and_preproc(pf_data_nok)
    get_separability(model, data_ok, data_nok, f_data_ok)
    dist_nok, bin_edges, proba_nok = get_distrib(model, data_nok)
    idx=10
    print(proba_nok.shape)
    pba = proba_nok[idx,0]
    save_trace(f"trace_{pba:.6f}", data_nok, idx)
    print(proba_nok[idx])
    # 
    if False:
        dist_nok, bin_edges, proba_nok = get_distrib(model, data_nok)
        plot_shower_trigged(data_nok, proba_nok[:,0])
        dist_ok, bin_edges, proba_ok = get_distrib(model, data_ok)
        plot_shower_trigged(data_ok, proba_ok[:, 0])

def plot_critere_1():
    a_perf = np.array([[715, 0.866], [300, 0.882]])
    plt.figure()
    plt.title("Diagramme: Indice de séparabilité, taux d'inférence\nData set ICRC")
    plt.grid()
    # plt.semilogy(a_perf[0,0], a_perf[0,1], 'd',label=f"ConvNet 2 couches")
    # plt.semilogy(a_perf[1,0], a_perf[1,1], 's',label=f"ConvNet 3 couches")
    plt.plot(a_perf[0,0], a_perf[0,1], 'd',label=f"ConvNet 2 couches")
    plt.plot(a_perf[1,0], a_perf[1,1], 's',label=f"ConvNet 3 couches")
    plt.legend()
    plt.ylim([0.8, 1])
    plt.xlabel("Hz, inférence par seconde sur dual Cortex A 53")
    plt.ylabel("Indice séparabilité")

def plot_critere_2():
    eff_txevt_2l = icrc_perfo_all('trigger2l.keras')
    eff_txevt_3l = icrc_perfo_all('trigger_icrc_80_acc96.keras')
    plt.figure()
    plt.title("Diagramme: Efficacité, taux de message\nHypothèse le bruit de fond passe le pré-trigger à 1kHz")
    plt.grid()
    # plt.semilogy(a_perf[0,0], a_perf[0,1], 'd',label=f"ConvNet 2 couches")
    # plt.semilogy(a_perf[1,0], a_perf[1,1], 's',label=f"ConvNet 3 couches")
    plt.plot(eff_txevt_2l[:,0], eff_txevt_2l[:,1],label=f"ConvNet 2 couches")
    plt.plot(eff_txevt_3l[:,0], eff_txevt_3l[:,1],label=f"ConvNet 3 couches")
    plt.axhline(250 , color='red', label='Limite capacité de transmission')
    plt.legend()
    #plt.ylim([0.8, 1])
    plt.ylabel("Hz\n taux d'événement à transmettre par seconde")
    plt.xlabel("Efficacité\nVaut 1 pour aucune perte d'événement gerbe cosmique")
    
if __name__ == '__main__':
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    #icrc_perfo()
    #concatenate_test_ok()
    #icrc_perfo_all()
    #plot_critere_1()
    #plot_critere_2()
    test_quant()
    # 
    plt.show()
