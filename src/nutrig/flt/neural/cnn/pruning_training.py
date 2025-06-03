"""
Created on 1 nov. 2024

@author: jcolley
"""

import time

import numpy as np
#from tensorflow import keras
#import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras
#import tensorflow_model_optimization as tfmot


import matplotlib.pyplot as plt
from rshower.basis.traces_event import Handling3dTraces

import nutrig.eval.cnn_ds_icrc as cnn_ds_icrc
import nutrig.eval.dataset as dataset
import nutrig.flt.neural.cnn.cnn_2layers as cnn_2l

## GLOBAL
traindir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
G_datadir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
quant = 2**13

# f_model = G_datadir + "template_wpp_2l_150.keras"
# f_model = G_datadir + "flt_2l_240920_150_ref.keras"
f_model = G_datadir + "flt_cnn_2l_240930_150.keras"
f_model = G_datadir + "flt_cnn_2l_241119_3.keras"
f_test_ok = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
f_test_nok = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
f_train_ok = G_datadir + "sig_dataset_nutrig_gp13_train_seed_300.npz"
f_train_nok = G_datadir + "bkg_dataset_nutrig_gp13_train_seed_300.npz"


def get_data_template(f_data, normalize=True):
    """

    :param f_data: (nb trace, 1024, 3)
    :param normalize:
    """

    # data["traces"].shape
    # (10000, 3, 1024)
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
    f_data = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
    f_proba = "sig_proba_cnn_nutrig_gp13_test_seed_300"
    write_prob_file(f_model, f_data, f_proba)


def get_proba_template_bkg():
    f_data = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
    f_data = G_datadir + "bkg_dataset_nutrig_gp13_th1_55_th2_35_test_seed_300.npz"
    f_proba = "bkg_proba_cnn_nutrig_gp13_test_seed_300"
    f_proba = "bkg_proba_cnn_nutrig_gp13_th1_55_th2_35_test_seed_300"
    write_prob_file(f_model, f_data, f_proba)


def get_sepabability_template():
    f_data_ok = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
    f_data_nok = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
    data_ok = get_data_template(f_data_ok)
    data_nok = get_data_template(f_data_nok)
    model = keras.models.load_model(f_model)
    cnn_ds_icrc.get_separability(model, data_ok, data_nok, f_data_ok.split("/")[-1])


def plot_dataset_snr():
    bins = [1, 4, 5, 6, 7, 8, 9, 10, 13, 17, 20, 30, 40, 50, 100]
    hsig = get_h3tr_template(f_test_ok, "Signal test")
    hbkg = get_h3tr_template(f_test_nok, "Background test")
    dataset.get_histo_snr(hbkg, hsig, "Test Template", bins, sigma=10)
    hsig = get_h3tr_template(f_train_ok, "Signal training")
    hbkg = get_h3tr_template(f_train_nok, "Background training")
    dataset.get_histo_snr(hbkg, hsig, "Training Template", bins, sigma=10)
    # dataset.get_histo_snr(hbkg, hsig, "Test Template2", sigma=10)


def get_sepabability_snr_template():
    """
    quick and dirty separability index versus SNR
    """
    f_data_ok = G_datadir + "sig_dataset_nutrig_gp13_test_seed_300.npz"
    f_data_nok = G_datadir + "bkg_dataset_nutrig_gp13_test_seed_300.npz"
    model = keras.models.load_model(f_model)
    #
    l_snr = [4, 5, 6, 7, 9, 10]
    # l_snr = [4,10]
    l_is = []
    l_nb_trace = []
    sigma_noise = 10
    for snr in l_snr:
        threshold = snr * sigma_noise
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
        # trok.plot_all_traces_as_image()
        data_ok = np.swapaxes(trok.traces, 1, 2) / quant
        #
        data_nok = get_data_template(f_data_nok, False)
        trnok = Handling3dTraces("Background only")
        trnok.init_traces(np.swapaxes(data_nok, 1, 2), f_samp_mhz=500)
        trnok.remove_trace_low_signal(threshold)
        nb_nok = trnok.get_nb_trace()
        # trnok.plot_all_traces_as_image()
        data_nok = np.swapaxes(trnok.traces, 1, 2) / quant
        res = cnn_ds_icrc.get_separability(
            model, data_ok, data_nok, f_data_ok.split("/")[-1], False
        )
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
    plt.plot(l_snr, a_nb[:, 0], label="Signal")
    plt.plot(l_snr, a_nb[:, 1], label="Background")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.ylabel("Number of traces")
    plt.xlabel("SNR threshold\n(sigma noise 10 ADU, signal max ||trace x,y,z||)")


def load_and_training():
    x_train, y_train = cnn_2l.datatset_template()
    training_cnn(x_train, y_train, G_datadir, "flt_cnn_2l_241119", regul=0, epochs=3)

def training_cnn(x_train, y_train, traindir, tf_mod_file="flt_2l_prun", epochs=100, regul=0):
    # x_train = x_train[:,:,:2]
    nb_trace = x_train.shape[0]
    nb_axis = x_train.shape[2]
    nb_sple = x_train.shape[1]
    print(x_train.shape)
    model = keras.Sequential(
        [
            keras.Input(shape=(nb_sple, nb_axis)),
            keras.layers.Conv1D(
                32,
                kernel_size=(11),
                padding="same",
                kernel_regularizer=keras.regularizers.l2(regul),
                activation="relu",
            ),
            keras.layers.MaxPooling1D(pool_size=(4)),
            keras.layers.Dropout(0.3),
            keras.layers.Conv1D(
                32,
                kernel_size=(11,),
                padding="same",
                kernel_regularizer=keras.regularizers.l2(regul),
                activation="relu",
            ),
            keras.layers.MaxPooling1D(pool_size=(4)),
            #
            # layers.Dropout(0.3),
            # layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            # layers.MaxPooling1D(pool_size=(2)),
            #
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.summary()

    batch_size = 128

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )
    ####
    array_loss_accu = np.empty((5, epochs), dtype=np.float32)

    array_loss_accu[0] = history.epoch
    array_loss_accu[1] = history.history["loss"]
    array_loss_accu[2] = history.history["val_loss"]
    array_loss_accu[3] = history.history["accuracy"]
    array_loss_accu[4] = history.history["val_accuracy"]
    np.save("perf_training", array_loss_accu)

    ####
    pnf_model = traindir + tf_mod_file + f"_{epochs}"
    pnf_model_keras = pnf_model + ".keras"
    # OLD model.save(pnf_model_keras)
    keras.models.save_model(model, pnf_model_keras)
    # converter = tf.lite.TFLiteConverter.from_saved_model(traindir)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    # tflite_model = converter.convert()
    # Save the model.
    # with open(pnf_model+'.tflite', 'wb') as f:
    #     f.write(tflite_model)

    plt.figure()
    plt.title(f"Loss function CNN 2 layers for {nb_trace} traces.")
    plt.plot(history.epoch, np.array(history.history["loss"]), label="Train loss")
    plt.plot(history.epoch, np.array(history.history["val_loss"]), label="Validation loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (binary crossentropy)")
    # plt.savefig(f"CNN_2_layer_loss_{tf_mod_file}_{nb_trace}")
    plt.figure()
    accuracy = history.history["accuracy"][-1] * 100
    plt.title(f"Accuracy value, CNN 2 layers for {nb_trace} traces. {accuracy:.1f}%")
    plt.plot(history.epoch, np.array(history.history["accuracy"]), label="Train accuracy")
    plt.plot(history.epoch, np.array(history.history["val_accuracy"]), label="Validation accuracy")
    plt.grid()
    plt.legend()
    # plt.title(str(int(np.ceil(history.history['accuracy'][-1] * 100))) + '%')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.savefig(f"CNN_2_layer_accuracy_{tf_mod_file}_{nb_trace}")

    score = model.evaluate(x_train, y_train, verbose=0)

    predictions = model.predict(x_train)
    print(predictions)
    print(y_train)
    print("Train loss:", score[0])
    print("Train accuracy:", score[1])


def pruning_training():
    """
    from https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras#fine-tune_pre-trained_model_with_pruning
    """
    model = keras.models.load_model(f_model)
    x_train, y_train = cnn_2l.datatset_template()
    import tensorflow_model_optimization as tfmot
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    
    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 2
    validation_split = 0.1  # 10% of training set will be used for validation set.

    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step
        )
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model_for_pruning.summary()


if __name__ == "__main__":

    # get_proba_template_sig()
    # get_sepabability_template()
    #get_sepabability_snr_template()
    # plot_dataset_snr()
    #
    #load_and_training()
    pruning_training()
    plt.show()
