"""
Author : Sandra Le Coz

Identification of air-shower radio pulses for the GRAND online trigger
S. Le Coz* and G. Collaboration

ICRC, July 2023, https://pos.sissa.it/444/224/

Trigger CNN

Refactoring for only training and pre-processing

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

nbant = 3
expsize = 1024
# packdir='/sps/grand/slecoz/ICRC23pack/'
traindir = '/home/jcolley/projet/grand_wk/data/npy/'
testdir = traindir

backg_train = np.load(traindir + 'day_backg_train.npy')
simu_train = np.load(traindir + 'day_simu_train.npy') 


def stats(data):
    std = np.std(data, axis=1)
    maxi = np.max(abs(data), axis=1)
    maxipos = np.argmax(abs(data), axis=1)
    maxistd = maxi / std
    return std, maxistd, maxipos


def plotstat(stat, xlabel, title, minimum=-1, maximum=-1):
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
    plt.show()

# preprocess


# simu_train
stdsimu, maxistdsimu, maxipossimu = stats(simu_train)

allmaxipossimu = np.sum((maxipossimu < 100) | (maxipossimu > (924)), axis=1)  # if true, to banish
simu_train = simu_train[(allmaxipossimu == 0)]

stdsimu, maxistdsimu, maxipossimu = stats(simu_train)

# backg_train
stdbackg, maxistdbackg, maxiposbackg = stats(backg_train)

allmaxiposbackg = np.sum((maxiposbackg < 100) | (maxiposbackg > (924)), axis=1)  # if true, to banish
backg_train = backg_train[(allmaxiposbackg == 0)]

stdbackg, maxistdbackg, maxiposbackg = stats(backg_train)

# make dataset

input_shape = (expsize, nbant)

print(len(simu_train), len(backg_train))
mini = np.min((len(simu_train), len(backg_train)))

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
liste = np.arange(mini * 2)
np.random.shuffle(liste)
xdata = xdata[liste]
ydata = ydata[liste]
print(ydata[0:999])
print(ydata[-999:])

# here!!
x_train = xdata
y_train = ydata

epochs = 80
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
model.save(traindir + f'trigger_icrc_{epochs}.keras')
plt.plot(history.epoch, np.array(history.history['loss']), label='Train loss')
plt.plot(history.epoch, np.array(history.history['val_loss']), label='Validation loss')
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss (binary crossentropy)')
plt.savefig('ICRC_loss')
plt.show()
plt.plot(history.epoch, np.array(history.history['accuracy']), label='Train accuracy')
plt.plot(history.epoch, np.array(history.history['val_accuracy']), label='Validation accuracy')
plt.grid()
plt.legend()
plt.title(str(int(np.ceil(history.history['accuracy'][-1] * 100))) + '%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('ICRC_accuracy')
plt.show()

score = model.evaluate(x_train, y_train, verbose=0)

predictions = model.predict(x_train)
print(predictions)
print(y_train)
print("Train loss:", score[0])
print("Train accuracy:", score[1])
