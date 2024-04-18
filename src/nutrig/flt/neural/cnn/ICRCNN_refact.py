"""
Author : Sandra Le Coz

Identification of air-shower radio pulses for the GRAND online trigger
S. Le Coz* and G. Collaboration

ICRC, July 2023, https://pos.sissa.it/444/224/

Trigger CNN

Refactoring for only training and pre-processing

"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers


nbant = 3
expsize = 1024
# packdir='/sps/grand/slecoz/ICRC23pack/'
traindir = '/home/jcolley/projet/grand_wk/data/npy/'
testdir = traindir


def stats(data):
    '''
    return for axis 1, ie sample of trace
      * standard dev
      * snr
      * index of absolute extremum
    
    :param data: float(nb_trace, nb_sample, nb_axis)
    
    :return: float(nb_trace, nb_axis)
    '''
    print("stats: ", data.shape)
    std = np.std(data, axis=1)
    maxi = np.max(abs(data), axis=1)
    maxipos = np.argmax(abs(data), axis=1)
    maxistd = maxi / std
    print(std.shape, maxistd.shape, maxipos.shape)
    print("stats std[:2]: ", std[:2]) 
    print("stats snr[:2]", maxistd[:2])
    return std, maxistd, maxipos


def plotstat(stat, xlabel, title, minimum=-1, maximum=-1):
    plt.figure()
    if minimum == -1:
        minimum = np.min(stat)
    if maximum == -1:
        maximum = np.max(stat)    
    plt.hist(stat[:, 0], histtype='step', bins=20, range=(minimum, maximum), linewidth=1)
    plt.hist(stat[:, 1], histtype='step', bins=20, range=(minimum, maximum), linewidth=1)
    plt.hist(stat[:, 2], histtype='step', bins=20, range=(minimum, maximum), linewidth=1)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(['X', 'Y', 'Z'])
    plt.savefig('ICRC_' + title[0])
    

def remove_pic_near_border(data, marge=100):
    '''
    remove traces near border with 'marge' samples
    
    :param data: float(nb_trace, nb_sample, nb_axis)
    :param marge: int
    '''
    nb_in = data.shape[0]
    high_marge = data.shape[1] - marge 
    _, _, maxipossimu = stats(data)
    # if true, to banish
    # | bitwise or of x and y
    # https://docs.python.org/3.8/library/stdtypes.html#bitwise-operations-on-integer-types
    allmaxipossimu = np.sum((maxipossimu < marge) | (maxipossimu > high_marge), axis=1)
    print(allmaxipossimu.shape)
    print(allmaxipossimu.dtype)
    #print(allmaxipossimu == 0)
    data = data[(allmaxipossimu == 0)]
    nb_out = data.shape[0]
    print(f"PREPROC : remove {nb_in-nb_out} traces")
    return data


def create_dataset_for_keras(data_ok, data_nok):
    d_shape = data_ok.shape
    expsize = d_shape[1]
    nbant = d_shape[2]
    mini = np.min((len(data_ok), len(data_nok)))
    xdata = np.zeros((mini * 2, expsize, nbant))
    xdata[:mini] = data_nok[:mini]
    xdata[mini:] = data_ok[:mini]
    ydata = np.zeros((mini * 2))
    ydata[:mini] = 0
    ydata[mini:] = 1
    return xdata, ydata


def save_trace(f_name, traces, idx):
    trace = traces[idx].T
    assert trace.shape == (nbant, expsize)
    np.savetxt(f_name + f"_{idx}.txt", trace, '%f', ',')
 

def icrc_training():
    #
    # ============================ MAIN
    #
    backg_train = np.load(traindir + 'day_backg_train.npy')
    simu_train = np.load(traindir + 'day_simu_train.npy') 
    
    # preprocess
    print("==================== preprocess")
    
    # simu_train
    # stdsimu, maxistdsimu, maxipossimu = stats(simu_train)
    # # if true, to banish
    # allmaxipossimu = np.sum((maxipossimu < 100) | (maxipossimu > (924)), axis=1)
    # simu_train = simu_train[(allmaxipossimu == 0)]
    simu_train = remove_pic_near_border(simu_train)
    stdsimu, maxistdsimu, maxipossimu = stats(simu_train)
    
    # backg_train
    # stdbackg, maxistdbackg, maxiposbackg = stats(backg_train)
    # # if true, to banish
    # allmaxiposbackg = np.sum((maxiposbackg < 100) | (maxiposbackg > (924)), axis=1)  
    # backg_train = backg_train[(allmaxiposbackg == 0)]
    backg_train = remove_pic_near_border(backg_train)
    stdbackg, maxistdbackg, maxiposbackg = stats(backg_train)
    
    # make dataset
    # same number of ok, nok traces
    #
    print("==================== make dataset")
    input_shape = (expsize, nbant)
    print(len(simu_train), len(backg_train))
    mini = np.min((len(simu_train), len(backg_train)))
    mini = 3000
    print(f"============= {mini} ==============")
    print(len(maxistdsimu[:mini]), len(maxistdbackg[:mini]))
    print(np.min((np.min(maxistdsimu[:mini]), np.min(maxistdbackg[:mini]))))
    minimini = np.min((np.min(maxistdsimu[:mini]), np.min(maxistdbackg[:mini])))
    maximaxi = np.max((np.max(maxistdsimu[:mini]), np.max(maxistdbackg[:mini])))
    plotstat(maxistdsimu[:mini], 'Trace maximum [std unit]', 'Air shower train dataset', minimum=minimini, maximum=maximaxi)
    plotstat(maxistdbackg[:mini], 'Trace maximum [std unit]', 'Background train dataset', minimum=minimini, maximum=maximaxi)
    xdata = np.zeros((mini * 2, expsize, nbant))
    xdata[:mini] = backg_train[:mini]
    xdata[mini:] = simu_train[:mini]
    ydata = np.zeros((mini * 2))
    ydata[:mini] = 0
    ydata[mini:] = 1
    
    # shuffle
    print("==================== shuffle")
    liste = np.arange(mini * 2)
    np.random.shuffle(liste)
    xdata = xdata[liste]
    ydata = ydata[liste]
    # print(ydata[0:999])
    # print(ydata[-999:])
    
    # sys.exit(-1)
    
    # here!!
    x_train = xdata
    y_train = ydata
    
    epochs = 100
    # regul=0.002
    regul = 0
    
    quant = 2 ** 13
    x_train = x_train / quant
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            layers.MaxPooling1D(pool_size=(2)),
            layers.Dropout(0.5),
            layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            layers.MaxPooling1D(pool_size=(2)),
            layers.Dropout(0.5),
            layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            layers.MaxPooling1D(pool_size=(2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
            
        ]
    )
    
    model.summary()
    
    batch_size = 128
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save(traindir + f'trigger_icrc2_{epochs}.keras')
    plt.figure()
    plt.title(f'Loss function for {mini} background traces.')
    plt.plot(history.epoch, np.array(history.history['loss']), label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']), label='Validation loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (binary crossentropy)')
    plt.savefig(f'ICRC_loss_{mini}')
    plt.figure()
    accuracy = history.history['accuracy'][-1]*100
    plt.title(f'Accuracy value for {mini} background traces. {accuracy:.1f}%')
    plt.plot(history.epoch, np.array(history.history['accuracy']), label='Train accuracy')
    plt.plot(history.epoch, np.array(history.history['val_accuracy']), label='Validation accuracy')
    plt.grid()
    plt.legend()
    #plt.title(str(int(np.ceil(history.history['accuracy'][-1] * 100))) + '%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'ICRC_accuracy_{mini}')
    
    score = model.evaluate(x_train, y_train, verbose=0)
    
    predictions = model.predict(x_train)
    print(predictions)
    print(y_train)
    print("Train loss:", score[0])
    print("Train accuracy:", score[1])
   
    
if __name__ == '__main__':
    #
    # https://keras.io/examples/keras_recipes/reproducibility_recipes/
    #
    #
    
    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) `tensorflow` random seed
    # 3) `python` random seed
    keras.utils.set_random_seed(813)
    
    # This will make TensorFlow ops as deterministic as possible, but it will
    # affect the overall performance, so it's not enabled by default.
    # `enable_op_determinism()` is introduced in TensorFlow 2.9.
    # tf.config.experimental.enable_op_determinism()
    
    icrc_training()
    # 
    plt.show()
