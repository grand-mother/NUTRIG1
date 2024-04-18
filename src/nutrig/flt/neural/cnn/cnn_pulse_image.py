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
traindir = '/home/jcolley/projet/grand_wk/data/npy/'
testdir = traindir
 

def training():
    #
    # ============================ MAIN
    #
    backg_train = np.load(traindir + 'day_backg_train.npy')
    simu_train = np.load(traindir + 'day_simu_train.npy') 
    
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
    min_nb = 3000
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
    y_train = ydata
    
    epochs = 100
    # regul=0.002
    regul = 0
    
    quant = 2 ** 13
    x_train = x_train / quant
    
    model = keras.Sequential(
        [   keras.Input(shape=(nb_sple, nb_axis)),
            layers.Conv1D(32, kernel_size=( 11), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            layers.MaxPooling1D(pool_size=(4)),
            layers.Dropout(0.3),
            layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            layers.MaxPooling1D(pool_size=(4)),
            #layers.Dropout(0.3),
            #layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
            #layers.MaxPooling1D(pool_size=(2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
            
        ]
    )
    
    model.summary()
    
    batch_size = 128
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    pnf_model = traindir + 'test2_fmt' 
    pnf_model_keras = pnf_model+'.keras'
    model.save(pnf_model_keras)
    #converter = tf.lite.TFLiteConverter.from_saved_model(traindir)
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_types = [tf.float16]    
    #tflite_model = converter.convert()
    # Save the model.
    # with open(pnf_model+'.tflite', 'wb') as f:
    #     f.write(tflite_model)

    plt.figure()
    plt.title(f'Loss function CNN 2 layers for {min_nb} background traces.')
    plt.plot(history.epoch, np.array(history.history['loss']), label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']), label='Validation loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (binary crossentropy)')
    plt.savefig(f'CNN_2_layer_loss_{min_nb}')
    plt.figure()
    accuracy = history.history['accuracy'][-1]*100
    plt.title(f'Accuracy value, CNN 2 layers for {min_nb} background traces. {accuracy:.1f}%')    
    plt.plot(history.epoch, np.array(history.history['accuracy']), label='Train accuracy')
    plt.plot(history.epoch, np.array(history.history['val_accuracy']), label='Validation accuracy')
    plt.grid()
    plt.legend()
    #plt.title(str(int(np.ceil(history.history['accuracy'][-1] * 100))) + '%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'CNN_2_layer_accuracy_{min_nb}')
    
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
    keras.utils.set_random_seed(812)
    
    # This will make TensorFlow ops as deterministic as possible, but it will
    # affect the overall performance, so it's not enabled by default.
    # `enable_op_determinism()` is introduced in TensorFlow 2.9.
    # tf.config.experimental.enable_op_determinism()

    training()
    # 
    plt.show()
