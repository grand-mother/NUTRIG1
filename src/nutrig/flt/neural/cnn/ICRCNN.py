"""
Author : Sandra Le Coz

Identification of air-shower radio pulses for the GRAND online trigger
S. Le Coz* and G. Collaboration

ICRC, July 2023, https://pos.sissa.it/444/224/

Trigger CNN

"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
from tensorflow import keras

noZ=0
day=1
nbant=3
expsize=1024
#packdir='/sps/grand/slecoz/ICRC23pack/'
traindir='datasetsICRC23/train/'
testdir='datasetsICRC23/test/'

#backg_train=np.load(traindir+'day_backgpack.npy')
if day:
    backg_train=np.load(traindir+'day_backgpack_doublepic_3sigma_cpt14687.npy') #day
    #backg_train=np.load(traindir+'day_backgpack_doublepic_3sigma_cpt14761_morerand.npy')
    simu_train=np.load(traindir+'day_inserted_simupack_doublepic3_3+_train.npy') 
else:    
    backg_train=np.load(traindir+'night_backgpack_doublepic_3sigma_cpt16193.npy') #night
    simu_train=np.load(traindir+'night_inserted_simupack_doublepic3_3+_train.npy') 
#simu_train=np.load(traindir+'day_inserted_simupack_1+.npy')


#backg_test=np.load(testdir+'day_backgpack.npy')
#simu_test=np.load(testdir+'day_inserted_simupack_5.npy')


def stats(data):
    std=np.std(data,axis=1)
    maxi=np.max(abs(data),axis=1)
    maxipos=np.argmax(abs(data),axis=1)
    maxistd=maxi/std
    return std,maxistd,maxipos


def plotstat(stat,xlabel,title,minimum=-1,maximum=-1,save=0):
   if minimum==-1:
       minimum=np.min(stat)
   if maximum==-1:
       maximum=np.max(stat)    
   plt.hist(stat[:,0],histtype='step',bins=20,range=(minimum,maximum),linewidth=1)
   plt.hist(stat[:,1],histtype='step',bins=20,range=(minimum,maximum),linewidth=1)
   plt.hist(stat[:,2],histtype='step',bins=20,range=(minimum,maximum),linewidth=1)
   plt.xlabel(xlabel)
   plt.title(title)
   plt.legend(['X','Y','Z'])
   plt.savefig('ICRC_'+title[0])
   plt.show()

#preprocess



#simu_train
stdsimu,maxistdsimu,maxipossimu=stats(simu_train)
'''plotstat(maxipossimu,'Maximum position [time sample number]','Simulation')
plotstat(maxistdsimu,'Maximum [std unit]','Simulation')
plotstat(stdsimu,'Standard deviation','Simulation')'''

allmaxipossimu=np.sum((maxipossimu<100) | (maxipossimu>(924)), axis=1) #if true, to banish
simu_train=simu_train[(allmaxipossimu==0)]

stdsimu,maxistdsimu,maxipossimu=stats(simu_train)
'''plotstat(maxipossimu,'Maximum position [time sample number]','Simulation')
plotstat(maxistdsimu,'Maximum [std unit]','Simulation')
plotstat(stdsimu,'Standard deviation','Simulation')'''




#backg_train
stdbackg,maxistdbackg,maxiposbackg=stats(backg_train)
'''plotstat(maxiposbackg,'Maximum position [time sample number]','Background')
plotstat(maxistdbackg,'Maximum [std unit]','Background')
plotstat(stdbackg,'Standard deviation','Background')'''

allmaxiposbackg=np.sum((maxiposbackg<100) | (maxiposbackg>(924)), axis=1) #if true, to banish
backg_train=backg_train[(allmaxiposbackg==0)]

stdbackg,maxistdbackg,maxiposbackg=stats(backg_train)
'''plotstat(maxiposbackg,'Maximum position [time sample number]','Background')
plotstat(maxistdbackg,'Maximum [std unit]','Background')
plotstat(stdbackg,'Standard deviation','Background')'''



'''plt.hist(maxistdbackg[:,0]/maxistdbackg[:,2])
plt.show()
plt.hist(maxistdsimu[:,0]/maxistdsimu[:,2])
plt.show()


simu_test=np.load(testdir+'day_inserted_simupack_doublepic3_3_test.npy')
stdsimu,maxistdsimu,maxipossimu=stats(simu_test)
plotstat(maxistdsimu,'toto','toto')'''





#make dataset


input_shape = (expsize, nbant)
if noZ:
    input_shape = (expsize, 1)

print(len(simu_train),len(backg_train))
mini=np.min((len(simu_train),len(backg_train)))

print(len(maxistdsimu[:mini]),len(maxistdbackg[:mini]))
print(np.min((np.min(maxistdsimu[:mini]),np.min(maxistdbackg[:mini]))))
minimini=np.min((np.min(maxistdsimu[:mini]),np.min(maxistdbackg[:mini])))
maximaxi=np.max((np.max(maxistdsimu[:mini]),np.max(maxistdbackg[:mini])))
plotstat(maxistdsimu[:mini],'Trace maximum [std unit]','Air shower train dataset',minimum=minimini,maximum=maximaxi)
plotstat(maxistdbackg[:mini],'Trace maximum [std unit]','Background train dataset',minimum=minimini,maximum=maximaxi)


xdata=np.zeros((mini*2,expsize,nbant))
xdata[:mini]=backg_train[:mini]
xdata[mini:]=simu_train[:mini]

ydata=np.zeros((mini*2))
ydata[:mini]=0
ydata[mini:]=1


#shuffle
liste=np.arange(mini*2)
np.random.shuffle(liste)
xdata=xdata[liste]
ydata=ydata[liste]
print(ydata[0:999])
print(ydata[-999:])

#here!!
x_train=xdata
y_train=ydata

if noZ:
    x_train=xdata[:,:,2]




'''splitratio=1

#train/test split
trainsize=int(splitratio*mini*2)
testsize=mini-trainsize

x_train=xdata[:trainsize]
y_train=ydata[:trainsize]

if splitratio<1:
    x_test=xdata[trainsize:]
    y_test=ydata[trainsize:]
    print(len(y_train),len(y_test))

else:
'''





epochs=80
#regul=0.002
regul=0


quant=2**13
x_train=x_train/quant


if 1:
    if 1:

        from keras import layers

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
                layers.MaxPooling1D(pool_size=(2)),
                layers.Dropout(0.5),
                layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul), activation="relu"),
                layers.MaxPooling1D(pool_size=(2)),
                layers.Dropout(0.5),
                layers.Conv1D(32, kernel_size=(11,), padding='same', kernel_regularizer=keras.regularizers.l2(regul),  activation="relu"),
                layers.MaxPooling1D(pool_size=(2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(1,activation="sigmoid"),
                
            ]
        )


                


        model.summary()

        batch_size = 128

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


        history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        plt.plot(history.epoch, np.array(history.history['loss']),label = 'Train loss')
        plt.plot(history.epoch, np.array(history.history['val_loss']),label = 'Validation loss')
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss (binary crossentropy)')
        plt.savefig('ICRC_loss')
        plt.show()
        plt.plot(history.epoch, np.array(history.history['accuracy']),label = 'Train accuracy')
        plt.plot(history.epoch, np.array(history.history['val_accuracy']),label = 'Validation accuracy')
        plt.grid()
        plt.legend()
        plt.title(str(int(np.ceil(history.history['accuracy'][-1]*100)))+'%')
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


#pred on test
'''score = model.evaluate(x_test, y_test, verbose=0)
predictions = model.predict(x_test)
print(predictions)
print(y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])'''


#chromatography
'''simuclass=predictions[y_test==1]
backgclass=predictions[y_test==0]


xax=np.zeros(len(y_test))
size0=len(y_test[y_test==0])
size1=len(y_test[y_test==1])
print(size0,size1)
xax[y_test==0]=np.arange(size0)
xax[y_test==1]=np.arange(size0,size0+size1)

plt.scatter(xax,predictions,c='orange',alpha=0.2)
plt.grid()
plt.xlabel('<--- Background | Sorted instance ID | Air shower --->')
plt.ylabel('Prediction')
plt.show()'''







#relationship between trace features and predictions
'''print(np.shape(predictions),np.shape(y_test))
stdtest,maxistdtest,maxipostest=stats(x_test)


std_simu=stdtest[y_test==1]
maxistd_simu=maxistdtest[y_test==1]
maxipos_simu=maxipostest[y_test==1]
std_backg=stdtest[y_test==0]
maxistd_backg=maxistdtest[y_test==0]
maxipos_backg=maxipostest[y_test==0]


std_simuwellclass=stdtest[(predictions[:,0]>=0.5) & (y_test==1)]
maxistd_simuwellclass=maxistdtest[(predictions[:,0]>=0.5) & (y_test==1)]
maxipos_simuwellclass=maxipostest[(predictions[:,0]>=0.5) & (y_test==1)]
std_simubadclass=stdtest[(predictions[:,0]<0.5) & (y_test==1)]
maxistd_simubadclass=maxistdtest[(predictions[:,0]<0.5) & (y_test==1)]
maxipos_simubadclass=maxipostest[(predictions[:,0]<0.5) & (y_test==1)]





std_backgwellclass=stdtest[(predictions[:,0]<0.5) & (y_test==0)]
maxistd_backgwellclass=maxistdtest[(predictions[:,0]<0.5) & (y_test==0)]
maxipos_backgwellclass=maxipostest[(predictions[:,0]<0.5) & (y_test==0)]
std_backgbadclass=stdtest[(predictions[:,0]>=0.5) & (y_test==0)]
maxistd_backgbadclass=maxistdtest[(predictions[:,0]>=0.5) & (y_test==0)]
maxipos_backgbadclass=maxipostest[(predictions[:,0]>=0.5) & (y_test==0)]

'''


#chromatograhy std
meanpred=np.zeros(6)
accuracy=np.zeros(6)
pred=np.zeros(0)
stdpred=np.zeros(0)
for i in range(6):
    #simutestfile=np.load(testdir+'day_inserted_simupack_'+str(i+1)+'.npy')
    if i<5:
        if day:
            simu_test=np.load(testdir+'day_inserted_simupack_doublepic3_'+str(i+3)+'_test.npy')
        else:
            simu_test=np.load(testdir+'night_inserted_simupack_doublepic3_'+str(i+3)+'_test.npy')
    if i==5:
        if day:
            simu_test=np.load(testdir+'day_inserted_simupack_8+.npy')
        else:
            simu_test=np.load(testdir+'night_inserted_simupack_8+.npy')
    print(len(simu_test))
    stdsimu,maxistdsimu,maxipossimu=stats(simu_test)
    allmaxipossimu=np.sum((maxipossimu<100) | (maxipossimu>924), axis=1) #if true, to banish
    simu_test=simu_test[(allmaxipossimu==0)]
    stdsimu,maxistdsimu,maxipossimu=stats(simu_test)
    
    if noZ:
        simu_test=simu_test[:,:,2]
    score = model.evaluate(simu_test/quant, np.zeros(len(simu_test))+1, verbose=0)
    predictions = model.predict(simu_test/quant)
    accuracy[i]=len(predictions[predictions>=0.5])/len(predictions)
    stdpred=np.concatenate((stdpred,np.random.rand(len(predictions))+(1+np.zeros(len(predictions)))*(i+3)))
    pred=np.concatenate((pred,predictions[:,0]))
    meanpred[i]=np.mean(predictions)
    print(meanpred[i])
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
print(meanpred)
print(accuracy)

plt.scatter(stdpred,pred,c='orange',alpha=0.2)
for i in range(6):
    plt.plot(np.arange(i+3,i+5),(meanpred[i],meanpred[i]),c='red',linestyle='dotted')
    plt.plot(np.arange(i+3,i+5),(accuracy[i],accuracy[i]),c='red')
plt.xlabel('3D-SNR maximum')
plt.ylabel('Score')
plt.legend(['Prediction', 'Averaged prediction', 'Accuracy'])
plt.plot(np.arange(3,10),np.zeros(7)+0.5,c='black',linestyle='dotted')
plt.title('Air shower test dataset')
plt.grid()
plt.savefig('airshowerperf')
plt.show()    

airshowerstdpred=stdpred
print(len(airshowerstdpred),len(stdpred))


#chromatography std for backg
#backg_test=np.load(testdir+'day_backgpack.npy')
if day:
    #backg_test=np.load(testdir+'day_backgpack_doublepic_3sigma_cpt14664.npy')
    backg_test=np.load(testdir+'day_backgpack_doublepic_3sigma_cpt15094_morerand.npy')
else:
    #backg_test=np.load(testdir+'night_backgpack_doublepic_3sigma_cpt15589.npy')
    backg_test=np.load(testdir+'night_backgpack_doublepic_3sigma_cpt15890_morerand.npy')

stdbackg,maxistdbackg,maxiposbackg=stats(backg_test)
allmaxiposbackg=np.sum((maxiposbackg<100) | (maxiposbackg>924), axis=1) #if true, to banish
backg_test=backg_test[(allmaxiposbackg==0)]
stdbackg,maxistdbackg,maxiposbackg=stats(backg_test)

if noZ:
    backg_test=backg_test[:,:,2]
score = model.evaluate(backg_test/quant, np.zeros(len(backg_test)), verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
predictions = model.predict(backg_test/quant)

sortedpred=sorted(predictions)
print(sortedpred)
nreject90=int(0.9*len(predictions))
threshold90=sortedpred[nreject90]
print("threshold for 90% rejection backg:",nreject90,threshold90)
nreject99=int(0.99*len(predictions))
threshold99=sortedpred[nreject99]
print("threshold for 99% rejection backg:",nreject99,threshold99)

for i in range(6):
    airshower90=len(pred[(airshowerstdpred>=i+3) & (airshowerstdpred<i+4) & (pred>threshold90)])/len(pred[(airshowerstdpred>=i+3) & (airshowerstdpred<i+4)])
    airshower99=len(pred[(airshowerstdpred>=i+3) & (airshowerstdpred<i+4) & (pred>threshold99)])/len(pred[(airshowerstdpred>=i+3) & (airshowerstdpred<i+4)])
    print("air shower for 90%,99% rejection backg:",i+3,airshower90,airshower99)

stdbackg,maxistdbackg,maxiposbackg=stats(backg_test)
stdpred=maxistdbackg

stdpred[stdpred>=8]=np.random.rand(len(stdpred[stdpred>=8]))+8 #tu put them is the same bin

meanpred=np.zeros(6)
accuracy=np.zeros(6)

for i in range(6):
    meanpred[i]=np.mean(predictions[(np.max(stdpred,axis=1)>=i+3) & (np.max(stdpred,axis=1)<i+4)])
    accuracy[i]=len(predictions[(np.max(stdpred,axis=1)>=i+3) & (np.max(stdpred,axis=1)<i+4) & (predictions[:,0]<0.5)])/len(predictions[(np.max(stdpred,axis=1)>=i+3) & (np.max(stdpred,axis=1)<i+4)])

print(meanpred)
print(accuracy)

plt.scatter(np.max(stdpred,axis=1),predictions[:,0],c='orange',alpha=0.2)
for i in range(6):
    plt.plot(np.arange(i+3,i+5),(meanpred[i],meanpred[i]),c='red',linestyle='dotted')
    plt.plot(np.arange(i+3,i+5),(accuracy[i],accuracy[i]),c='red')
   
plt.xlabel('3D-trace maximum [std unit]')
plt.ylabel('Score')
plt.legend(['Prediction', 'Averaged prediction', 'Accuracy'])
plt.plot(np.arange(3,10),np.zeros(7)+0.5,c='black',linestyle='dotted')
plt.title('Background test dataset')
plt.grid()
plt.savefig('backgroundperf')
plt.show()    

print(len(airshowerstdpred),len(stdpred))

for i in range(3,9):

    plt.hist(predictions[:,0][np.max(stdpred,axis=1)>=i], bins=20,histtype='step',log=True,density=1)
    plt.hist(pred[airshowerstdpred>=i], bins=20,histtype='step',log=True,density=1)
    plt.xlabel('Prediction')
    plt.title('SNR = '+str(i))
    plt.legend(['Background','Air shower'])
    plt.ylim(1e-3,100)
    plt.savefig('histopred'+str(i)+'.png')
    plt.show()


















'''
for i in range(nbant):
    maxistd_simuwellclass_h,b=np.histogram(maxistd_simuwellclass[:,i],bins=20,range=(np.min(maxistd_simu),np.max(maxistd_simu)))
    maxistd_simu_h,b=np.histogram(maxistd_simu[:,i],bins=20,range=(np.min(maxistd_simu),np.max(maxistd_simu)))
    plt.plot(b[:-1][maxistd_simu_h>0],100*maxistd_simuwellclass_h[maxistd_simu_h>0]/maxistd_simu_h[maxistd_simu_h>0])

plt.vlines(np.max(maxistd_simu),0,100,colors='red')
plt.grid()
plt.legend(['X','Y','Z'])
plt.xlabel('Maximum [std unit]')
plt.ylabel('% of well classified air shower 3D-traces')
plt.savefig('wellclassified_airshower')
plt.show()

for i in range(nbant):
    maxistd_backgwellclass_h,b=np.histogram(maxistd_backgwellclass[:,i],bins=20,range=(np.min(maxistd_backg),np.max(maxistd_backg)))
    maxistd_backg_h,b=np.histogram(maxistd_backg[:,i],bins=20,range=(np.min(maxistd_backg),np.max(maxistd_backg)))
    plt.plot(b[:-1][maxistd_backg_h>0],100*maxistd_backgwellclass_h[maxistd_backg_h>0]/maxistd_backg_h[maxistd_backg_h>0])

plt.vlines(np.max(maxistd_backg),0,100,colors='red')
plt.grid()
plt.legend(['X','Y','Z'])
plt.xlabel('Maximum [std unit]')
plt.ylabel('% of well classified background 3D-traces')
plt.savefig('wellclassified_background')
plt.show()'''


'''
#threshold selection
nthreshold=10
nmaxistd=5
nsimu=np.zeros((nthreshold,nmaxistd+1))
nbackg=np.zeros((nthreshold,nmaxistd+1))
for i in range(nthreshold):
    threshold=i*0.1
    for j in range(nmaxistd):
        thismaxistd=j+2
        thismaxistd_nsimu=len(predictions[(y_test==1) & (np.max(maxistdtest,axis=1).astype('int')==thismaxistd)] )
        thismaxistd_nbackg=len(predictions[(y_test==0) & (np.max(maxistdtest,axis=1).astype('int')==thismaxistd)] )
        print(threshold,thismaxistd,thismaxistd_nsimu,thismaxistd_nbackg)
        if thismaxistd_nsimu >0:
            nsimu[i,j]=len(predictions[(predictions[:,0]>=threshold) & (y_test==1) & (np.max(maxistdtest,axis=1).astype('int')==thismaxistd) ]) /thismaxistd_nsimu
        if thismaxistd_nbackg>0:
            nbackg[i,j]=len(predictions[(predictions[:,0]>=threshold) & (y_test==0) & (np.max(maxistdtest,axis=1).astype('int')==thismaxistd) ]) /thismaxistd_nbackg
    
    thismaxistd_nsimu=len(predictions[(y_test==1) & (np.max(maxistdtest,axis=1).astype('int')>thismaxistd)] )
    thismaxistd_nbackg=len(predictions[(y_test==0) & (np.max(maxistdtest,axis=1).astype('int')>thismaxistd)] )
    if thismaxistd_nsimu>0:
        nsimu[i,-1]=len(predictions[(predictions[:,0]>=threshold) & (y_test==1) & (np.max(maxistdtest,axis=1).astype('int')>thismaxistd) ]) /thismaxistd_nsimu
    if thismaxistd_nbackg>0:
        nbackg[i,-1]=len(predictions[(predictions[:,0]>=threshold) & (y_test==0) & (np.max(maxistdtest,axis=1).astype('int')>thismaxistd) ]) /thismaxistd_nbackg
    



for j in range(nmaxistd+1):

    plt.plot(np.arange(nthreshold)*0.1,nbackg[:,j])

plt.grid()
plt.title('Background')
plt.xlabel('Selection threshold')
plt.ylabel('% of selected')
plt.legend(['2','3','4','5','6','7+'])
plt.savefig('selectionthreshold_background')
plt.show()

for j in range(nmaxistd+1):

    plt.plot(np.arange(nthreshold)*0.1,nsimu[:,j])

plt.grid()
plt.title('Air shower')
plt.xlabel('Selection threshold')
plt.ylabel('% of selected')
plt.legend(['2','3','4','5','6','7+'])
plt.savefig('selectionthreshold_airshower')
plt.show()

'''



