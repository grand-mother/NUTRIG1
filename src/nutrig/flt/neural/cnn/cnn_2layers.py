"""
Colley Jean-marc,

"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

import nutrig.flt.neural.cnn.ICRCNN_refact as ppd
import nutrig.flt.common.tools_trace as ttr

nb_axis = 3
expsize = 1024
# packdir='/sps/grand/slecoz/ICRC23pack/'


def trainin_cnn(x_train, y_train, traindir, tf_mod_file="flt_2l", epochs=100, regul=0):
    # x_train = x_train[:,:,:2]
    nb_trace = x_train.shape[0]
    nb_axis = x_train.shape[2]
    nb_sple = x_train.shape[1]
    assert nb_axis == 3
    print(x_train.shape)
    model = keras.Sequential(
        [
            keras.Input(shape=(nb_sple, nb_axis)),
            layers.Conv1D(
                32,
                kernel_size=(11),
                padding="same",
                kernel_regularizer=keras.regularizers.l2(regul),
                activation="relu",
            ),
            layers.MaxPooling1D(pool_size=(4)),
            layers.Dropout(0.3),
            layers.Conv1D(
                32,
                kernel_size=(11,),
                padding="same",
                kernel_regularizer=keras.regularizers.l2(regul),
                activation="relu",
            ),
            layers.MaxPooling1D(pool_size=(4)),
            #
            # layers.Dropout(0.3),
            # layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            # layers.MaxPooling1D(pool_size=(2)),
            #
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
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
    model.save(pnf_model_keras)
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


def training_with_icrc():
    traindir = "/home/jcolley/projet/grand_wk/data/npy/dataset_icrc/"
    print("\n\n=====================  training_with_icrc")
    #
    # ============================ MAIN
    #
    backg_train = np.load(traindir + "day_backg_train.npy")
    simu_train = np.load(traindir + "day_simu_train.npy")
    print(simu_train.shape)

    # preprocess
    print("==================== preprocess")
    simu_train = ppd.remove_pic_near_border(simu_train)
    stdsimu, maxistdsimu, maxipossimu = ppd.stats(simu_train)
    backg_train = ppd.remove_pic_near_border(backg_train)
    stdbackg, maxistdbackg, maxiposbackg = ppd.stats(backg_train)
    pe3 = ttr.PulsExtractor3D(np.swapaxes(simu_train, 1, 2))
    simu_train = pe3.get_pulse3d()
    # pe3.plot_pulse_selected(20,30)
    pe3 = ttr.PulsExtractor3D(np.swapaxes(backg_train, 1, 2))
    backg_train = pe3.get_pulse3d()
    # pe3.plot_pulse_selected(20, 40)
    plt.show()
    # make dataset
    # same number of ok, nok traces
    #
    print("==================== make dataset")
    nb_sple = simu_train.shape[2]
    nb_axis = simu_train.shape[1]
    nb_train = simu_train.shape[0]
    nb_back = backg_train.shape[0]
    min_nb = np.min([nb_train, nb_back])
    # min_nb = 3000
    xdata = np.zeros((min_nb * 2, nb_axis, nb_sple))
    xdata[:min_nb] = backg_train[:min_nb]
    xdata[min_nb:] = simu_train[:min_nb]
    ydata = np.zeros((min_nb * 2))
    ydata[:min_nb] = 0
    ydata[min_nb:] = 1

    # shuffle
    print("==================== shuffle")
    liste = np.arange(min_nb * 2)
    np.random.shuffle(liste)
    xdata = xdata[liste]
    ydata = ydata[liste]

    x_train = np.swapaxes(xdata, 1, 2).reshape(min_nb * 2, nb_sple, nb_axis, 1)
    print(x_train.shape)
    return
    y_train = ydata

    epochs = 100
    # regul=0.002
    regul = 0

    quant = 2**13
    x_train = x_train / quant
    trainin_cnn(x_train, y_train, traindir, "with_icrc", epochs=90)


def training_with_icrc_clean():
    traindir = "/home/jcolley/projet/grand_wk/data/npy/dataset_icrc/"
    print("\n\n=====================  training_with_icrc")
    #
    # ============================ MAIN
    #
    backg_train = np.load(traindir + "day_backg_train.npy")
    simu_train = np.load(traindir + "day_simu_train.npy")
    print(simu_train.shape)

    # preprocess
    print("==================== preprocess")
    simu_train = ppd.remove_pic_near_border(simu_train)
    stdsimu, maxistdsimu, maxipossimu = ppd.stats(simu_train)
    backg_train = ppd.remove_pic_near_border(backg_train)
    stdbackg, maxistdbackg, maxiposbackg = ppd.stats(backg_train)

    # make dataset
    # same number of ok, nok traces
    #
    print("==================== make dataset")
    nb_sple = simu_train.shape[2]
    nb_axis = simu_train.shape[1]
    nb_train = simu_train.shape[0]
    nb_back = backg_train.shape[0]
    min_nb = np.min([nb_train, nb_back])
    # min_nb = 3000
    xdata = np.zeros((min_nb * 2, nb_axis, nb_sple))
    xdata[:min_nb] = backg_train[:min_nb]
    xdata[min_nb:] = simu_train[:min_nb]
    ydata = np.zeros((min_nb * 2))
    ydata[:min_nb] = 0
    ydata[min_nb:] = 1

    # shuffle
    print("==================== shuffle")
    liste = np.arange(min_nb * 2)
    np.random.shuffle(liste)
    xdata = xdata[liste]
    ydata = ydata[liste]

    x_train = xdata
    print(x_train.shape)
    y_train = ydata

    epochs = 100
    # regul=0.002
    regul = 0

    quant = 2**13
    x_train = x_train / quant
    trainin_cnn(x_train, y_train, traindir, "with_icrc", epochs=90)


def datatset_template():
    """
    return:
        x_train: trace (trace, sample, axis)
        y_train: label 0 or 1
    """
    traindir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
    #
    # ============================ training_with_template
    #
    print("\n\n ============================ training_with_template")
    backg_train = np.load(traindir + "bkg_dataset_nutrig_gp13_train_seed_300.npz")["traces"]
    simu_train = np.load(traindir + "sig_dataset_nutrig_gp13_train_seed_300.npz")["traces"]
    print(simu_train.shape)
    # need move axis to have the same shape that ICRC
    backg_train = np.moveaxis(backg_train, 1, 2)
    simu_train = np.moveaxis(simu_train, 1, 2)
    print(np.isnan(simu_train).any())
    print(np.isnan(backg_train).any())
    print(np.mean(simu_train))
    print(np.std(simu_train))
    print(np.mean(backg_train))
    print(np.std(backg_train))
    # preprocess
    if False:
        simu_train = ppd.remove_pic_near_border(simu_train)
        stdsimu, maxistdsimu, maxipossimu = ppd.stats(simu_train)
        backg_train = ppd.remove_pic_near_border(backg_train)
        stdbackg, maxistdbackg, maxiposbackg = ppd.stats(backg_train)
    #
    print("==================== make dataset")
    nb_sple = simu_train.shape[2]
    nb_axis = simu_train.shape[1]
    nb_train = simu_train.shape[0]
    nb_back = backg_train.shape[0]
    min_nb = np.min([nb_train, nb_back])
    # min_nb = 4000
    xdata = np.zeros((min_nb * 2, nb_axis, nb_sple))
    xdata[:min_nb] = backg_train[:min_nb]
    xdata[min_nb:] = simu_train[:min_nb]
    ydata = np.zeros((min_nb * 2))
    ydata[:min_nb] = 0
    ydata[min_nb:] = 1
    # shuffle
    print("==================== shuffle")
    liste = np.arange(min_nb * 2)
    np.random.shuffle(liste)
    x_train = xdata[liste]
    y_train = ydata[liste]
    print("x_train.shape: ", x_train.shape)
    epochs = 100
    # regul=0.002
    regul = 0
    quant = 2**13
    x_train = x_train / quant
    return x_train, y_train


def training_with_template():
    traindir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
    #
    # ============================ training_with_template
    #
    print("\n\n ============================ training_with_template")
    backg_train = np.load(traindir + "bkg_dataset_nutrig_gp13_train_seed_300.npz")["traces"]
    simu_train = np.load(traindir + "sig_dataset_nutrig_gp13_train_seed_300.npz")["traces"]
    print(simu_train.shape)

    # need move axis to have the same shape that ICRC
    backg_train = np.moveaxis(backg_train, 1, 2)
    simu_train = np.moveaxis(simu_train, 1, 2)

    print(np.isnan(simu_train).any())
    print(np.isnan(backg_train).any())
    print(np.mean(simu_train))
    print(np.std(simu_train))
    print(np.mean(backg_train))
    print(np.std(backg_train))
    # preprocess
    if False:
        simu_train = ppd.remove_pic_near_border(simu_train)
        stdsimu, maxistdsimu, maxipossimu = ppd.stats(simu_train)
        backg_train = ppd.remove_pic_near_border(backg_train)
        stdbackg, maxistdbackg, maxiposbackg = ppd.stats(backg_train)
    #
    print("==================== make dataset")
    nb_sple = simu_train.shape[2]
    nb_axis = simu_train.shape[1]
    nb_train = simu_train.shape[0]
    nb_back = backg_train.shape[0]
    min_nb = np.min([nb_train, nb_back])
    # min_nb = 4000
    xdata = np.zeros((min_nb * 2, nb_axis, nb_sple))
    xdata[:min_nb] = backg_train[:min_nb]
    xdata[min_nb:] = simu_train[:min_nb]
    ydata = np.zeros((min_nb * 2))
    ydata[:min_nb] = 0
    ydata[min_nb:] = 1

    # shuffle
    print("==================== shuffle")
    liste = np.arange(min_nb * 2)
    np.random.shuffle(liste)
    x_train = xdata[liste]
    y_train = ydata[liste]

    print("x_train.shape: ", x_train.shape)

    epochs = 100
    # regul=0.002
    regul = 0

    quant = 2**13
    x_train = x_train / quant
    trainin_cnn(x_train, y_train, traindir, "flt_cnn_2l_240930", regul=0, epochs=100)


def training_with_template_2():
    traindir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_2.0/"
    #
    # ============================ training_with_template
    #
    print("\n\n ============================ training_with_template")
    f_bkg = traindir + "bkg_database_nutrig_v2_FILTERED_UNIFORM.npz"
    f_sig = traindir + "sig_database_nutrig_v2_FILTERED_UNIFORM.npz"
    backg_train = np.load(f_bkg)["traces"]
    simu_train = np.load(f_sig)["traces"]

    # need move axis to have the same shape that ICRC

    backg_train = np.swapaxes(backg_train, 1, 2)
    simu_train = np.swapaxes(simu_train, 1, 2)
    print("==================== shuffle")
    np.random.shuffle(backg_train)
    np.random.shuffle(simu_train)
    print("simu: ", simu_train.shape)
    print("back: ", backg_train.shape)

    print(np.isnan(simu_train).any())
    print(np.isnan(backg_train).any())
    print(np.mean(simu_train))
    print(np.std(simu_train))
    print(np.mean(backg_train))
    print(np.std(backg_train))

    # preprocess
    if False:
        simu_train = ppd.remove_pic_near_border(simu_train)
        stdsimu, maxistdsimu, maxipossimu = ppd.stats(simu_train)
        backg_train = ppd.remove_pic_near_border(backg_train)
        stdbackg, maxistdbackg, maxiposbackg = ppd.stats(backg_train)
    #
    print("==================== make dataset")
    nb_sple = simu_train.shape[2]
    nb_axis = simu_train.shape[1]
    nb_train = simu_train.shape[0]
    nb_back = backg_train.shape[0]
    min_nb = np.min([nb_train, nb_back])
    min_nb = 4000
    # save split data
    sig_test = f_sig.replace("nutrig_v2_FILTERED_UNIFORM", "shuffle_test")
    sig_test = sig_test.replace(".npz", "")
    print(sig_test)
    np.save(sig_test, simu_train[min_nb:])
    bkg_test = f_bkg.replace("nutrig_v2_FILTERED_UNIFORM", "shuffle_test")
    bkg_test = bkg_test.replace(".npz", "")
    np.save(bkg_test, backg_train[min_nb:])
    ###
    xdata = np.zeros((min_nb * 2, nb_axis, nb_sple))
    xdata[:min_nb] = backg_train[:min_nb]
    xdata[min_nb:] = simu_train[:min_nb]
    ydata = np.zeros((min_nb * 2))
    ydata[:min_nb] = 0
    ydata[min_nb:] = 1

    print("x_train.shape: ", ydata.shape)

    n_ep = 20
    # regul=0.002
    v_reg = 0

    quant = 2**13
    xdata = xdata / quant
    trainin_cnn(xdata, ydata, traindir, "flt_cnn_2l_2509_a", regul=v_reg, epochs=n_ep)


if __name__ == "__main__":
    #
    # https://keras.io/examples/keras_recipes/reproducibility_recipes/
    #
    #

    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) `tensorflow` random seed
    # 3) `python` random seed
    keras.utils.set_random_seed(815)

    # This will make TensorFlow ops as deterministic as possible, but it will
    # affect the overall performance, so it's not enabled by default.
    # `enable_op_determinism()` is introduced in TensorFlow 2.9.
    # tf.config.experimental.enable_op_determinism()

    # training_with_icrc_clean()
    training_with_template_2()
    #
    plt.show()
