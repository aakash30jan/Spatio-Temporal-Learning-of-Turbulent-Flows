#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------------------------
Supporting: Autoregressive Transformers for Data-Driven Spatio-Temporal Learning of Turbulent Flows
URL:        https://arxiv.org/abs/2209.08052
Author:     Aakash Patil, Jonathan Viquerat, Elie Hachem                             
Year:       March, 2023                                                
-------------------------------------------------------------------------------------------------
"""

"""
USAGE examples:  
python ./train.py 1 both 2 case2 1 2
python ./train.py 1 both 10 case1 1 2
python ./train.py 1 both 1000 case2 1 2
"""
import sys
import os

print("python train.py sampledBy useVar epochs DATASET TIN TNEXT")
print("python train.py INT nuTilde INT")
print("python train.py INT velocity INT")
sys.argv
total = len(sys.argv)
cmdargs = str(sys.argv)
print ("The total numbers of args passed to the script: %d " % total)
print ("Args list: %s " % cmdargs)

cmdargs = sys.argv
SAMPLEDBY = int(cmdargs[1])
useVar = cmdargs[2]
EPOCHS = int(cmdargs[3])
DATASET = cmdargs[4]
TIN = int(cmdargs[5])
TNEXT = int(cmdargs[6])
try:
    NORMED = cmdargs[7]
except:
    print("param NORMED not provided ")
    NORMED = 'notnormed'
TEND = TNEXT
print("sampledBy , useVar, epochs, DATASET, TIN, TNEXT ", SAMPLEDBY , useVar, EPOCHS, DATASET, TIN, TNEXT   )

if useVar=='nuTilde' or useVar=='velocity'or useVar=='both' or useVar=='all' or useVar=='levelSet' or useVar=='bothCellVals':
    print("Using useVar=",useVar)
else:
    print('Undefined useVar=',useVar)
    sys.exit()

sampledBy = SAMPLEDBY 
epochs = EPOCHS
dataset= DATASET  
if NORMED == 'normed':
    normAll = True
else:
    normAll = False

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten, MaxPool2D, Activation, UpSampling2D
from tqdm import tqdm
from functions import *

newlib_path = '../'
sys.path.insert(1, os.path.join(os.getcwd(),newlib_path))
useGPU = 0 #0 #1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[useGPU], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        tf.config.experimental.set_memory_growth(gpus[useGPU], True)
    except RuntimeError as e:
        print(" Visible devices must be set BEFORE GPUs have been initialized")
        print(e)

if normAll:
    addSuff='_normed'
else:
    addSuff=''
#####################################################

trainDirName = 'train_'+useVar+'_samp'+str(SAMPLEDBY)+'_ep'+str(EPOCHS)+'_'+DATASET+'_INOUT_'+str(TIN)+'_'+str(TNEXT)+addSuff

if not os.path.exists(trainDirName):
    os.mkdir(trainDirName)
os.chdir(trainDirName)


################## LOAD DATA ##################

if dataset== 'case1':
    try:
          datasetfile = '../../data/sampleData_case1.npy' 
          data = np.load(datasetfile,mmap_mode='r')
    except:
          datasetfile = '../../data/sampleData_case1.npy'
          data = np.load(datasetfile,mmap_mode='r')
    print("Loaded npy shape is: ", data.shape) #Nsamples, nx,ny, ch 
    takeData = data.shape[0]-1 #until index

elif dataset== 'case2':
    try:
          datasetfile = '../../data/sampleData_case2.npy' 
          data = np.load(datasetfile,mmap_mode='r')
    except:
          datasetfile = '../../data/sampleData_case2.npy'
          data = np.load(datasetfile,mmap_mode='r')
    print("Loaded npy shape is: ", data.shape) #Nsamples, nx,ny, ch 
    takeData = data.shape[0]-1 #until index
else:
    print("UNEXPECTED dataset==", dataset)
    sys.exit()

#####################################################
currX, currY = data.shape[1], data.shape[2]
sampledBy = SAMPLEDBY 
tin = TIN #1 #from tin
tnext = TNEXT #5#1  # next
data = data[:takeData][::sampledBy]
print("Sampled npy shape is: ", data.shape) #Nsamples, nx,ny, ch 

if normAll:
    print("Normalizing data ")
    data = doNorm_CHwise(data, minV=None, maxV=None)

if useVar=='nuTilde' :
    data = np.expand_dims(data[...,3] , axis = -1)  
elif useVar=='velocity':
    data = data[...,:2]  
elif useVar=='both':
    data = data[...,:3]  
elif useVar=='all':
    data = data[...,:]  
elif useVar=='levelSet':
    data = data[...,3:] 
elif useVar=='bothCellVals':
    data = data[...,1:]  
else:
    print('Undefined useVar=',useVar)
    sys.exit()

print("data.shape ", data.shape)

snapshots = data.shape[0]
ipseq, opseq = getTseq_inout(snapshots, tin, tnext)
data_LR  = data[ipseq]
if tin == 1:
    data_LR = np.squeeze(data_LR, axis=1)
data_HR  = data[opseq]  

print("data_HR.shape ", data_HR.shape)
print("data_LR.shape ", data_LR.shape)

if tin==1 and tnext==1:
    data_HR = np.squeeze(data_HR, axis=1)
    data_LR = np.squeeze(data_LR, axis=1)
    print("data_HR.shape ", data_HR.shape)
    print("data_LR.shape ", data_LR.shape)


"""
import matplotlib.pyplot as plt
plt.clf()
plt.imshow(data_HR[0,...,0])
plt.savefig('sampleHR_360x300_ch0.png')
try:
    plt.clf()
    plt.imshow(data_HR[0,...,1])
    plt.savefig('sampleHR_360x300_ch1.png')
except:
    pass
"""
#####################################################
## Resize/Reshape so that AutoEnc doesn't go burrrr
data_HR_wasModif = False
data_LR_wasModif = False

if 'Filling' in dataset:
    ResizeData = False
    newX,newY = 192,96    
else:
    ResizeData = True
    newX,newY = 192,128
if currX==192 and currY==128 :
    print("Resize not required as currX=newX and currY=newY")
    ResizeData = False
    newX,newY = 192,128    
if ResizeData:
    if data_HR.ndim == 5:
        data_HR_dims = data_HR.shape
        data_HR = data_HR.reshape(data_HR_dims[0]*data_HR_dims[1], data_HR_dims[2], data_HR_dims[3], data_HR_dims[4])
        data_HR_wasModif = True
    if data_LR.ndim == 5:
        data_LR_dims = data_LR.shape
        data_LR = data_LR.reshape(data_LR_dims[0]*data_LR_dims[1], data_LR_dims[2], data_LR_dims[3], data_LR_dims[4])
        data_LR_wasModif = True
    data_LR = tf.image.resize( data_LR, (newX,newY), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None ).numpy()
    data_LR_dims_new = data_LR.shape
    data_HR = tf.image.resize( data_HR, (newX,newY), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None ).numpy()
    data_HR_dims_new = data_HR.shape
    if data_HR_wasModif:
        data_HR = data_HR.reshape(data_HR_dims[0],data_HR_dims[1], data_HR_dims_new[1], data_HR_dims_new[2], data_HR_dims_new[3])
    if data_LR_wasModif:
        data_LR = data_LR.reshape(data_LR_dims[0],data_LR_dims[1], data_LR_dims_new[1], data_LR_dims_new[2], data_LR_dims_new[3])

    print("data_HR.shape ", data_HR.shape)
    print("data_LR.shape ", data_LR.shape)

"""
plt.clf()
plt.imshow(data_HR[0,...,0])
plt.savefig('sampleHR_192x128_ch0.png')

try:
    plt.clf()
    plt.imshow(data_HR[0,...,1])
    plt.savefig('sampleHR_192x128_ch1.png')
except:
    pass

for i in range(10):
    plt.clf()
    plt.imshow(data_HR[i,...,0])
    plt.savefig('sampleHR_192x128_ch0_i'+str(i)+'.png')
"""

#### Assert sample size
if data_HR.shape[0] != data_LR.shape[0]:
      print("Unequal Samples in LR and HR\n Exiting !")
      sys.exit();
#####################################################

inputShape = np.array(data_LR.shape) #data_HR
input_shape = (inputShape[1], inputShape[2], inputShape[3])

#model = tmodel_autoReg(model_convXformer,input_shape=input_shape, sampling_size=(24, 16), tend=2)
 
if 'Filling_' in dataset: 
    sampling_size=(24, 12)
else:
    sampling_size=(24, 16)

model = tmodel_autoReg_run(model_convXformer,input_shape=input_shape, sampling_size=sampling_size, tend=TEND)

print(model.summary())
loss_MSE = tf.keras.losses.MeanSquaredError()
model.compile(loss=loss_MSE, optimizer='adam')

epochs = EPOCHS
batch_size = 4 
trainValidRatio = 70

trainVal = int( np.ceil(snapshots*(trainValidRatio/100)) )
print(trainVal," is trainVal and split: data_LR[:trainVal], data_LR[trainVal:]", data_LR[:trainVal].shape, data_LR[trainVal:].shape)
print("leftover data.shape ", data.shape)

np.save("input_LR.npy", data_LR[trainVal:])
print("input_LR.shape ", data_LR[trainVal:].shape)
history = tf.keras.callbacks.History()
callbacksList = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=250 , verbose=0, mode='auto'),
tf.keras.callbacks.ModelCheckpoint('best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1),  history]

callbacksList = callbacksList[1:] #remove early stop for all

hist = model.fit(data_LR[:trainVal],data_HR[:trainVal],
             batch_size = batch_size,
             verbose=1,
             epochs = epochs,
             validation_data=(data_LR[trainVal:],data_HR[trainVal:]),shuffle=False ,
             callbacks=callbacksList )

import pickle
try:
    model.save('modelLastBest.h5')
except:
    print('Saving full model failed, saving last best weights instead ')
    model.save_weights("modelLastBest_weights.h5")

def save_obj(obj, name ):
    with open('obj_'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open('obj_' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(history.history, 'model_history')
history_dict = load_obj('model_history' )
outfile = open( 'model_history_dict.txt', 'w' )
for key, value in sorted( history_dict.items() ):
    outfile.write( str(key) + '\t' + str(value) + '\n' )
outfile.close()



predicted_HR = model.predict(data_LR[trainVal:])
expected_HR = data_HR[trainVal:]

print("predicted_HR.shape ", predicted_HR.shape)
print("expected_HR.shape ",expected_HR.shape )
np.save("expected_HR.npy", expected_HR)
np.save("predicted_HR.npy", predicted_HR)
print("saved expected_HR.npy  predicted_HR.npy")


#recySim
print("Performing recySim ")
inputLR_valid = data_LR[trainVal:]
predicted_HRRecy = model.predict( np.expand_dims( inputLR_valid[0], axis=0) )
for t in tqdm(range(1,inputLR_valid.shape[0])):
    currentPred = model.predict( np.expand_dims( predicted_HRRecy[t-1][-1], axis=0) )
    predicted_HRRecy = np.append( predicted_HRRecy, currentPred , axis=0)

print("predicted_HRRecy.shape ", predicted_HRRecy.shape)
np.save("predicted_HRRecy.npy", predicted_HRRecy)
print("saved predicted_HRRecy.npy")
np.save("input_LR.npy", inputLR_valid)
print(" saved input_LR.npy with shape ", inputLR_valid.shape)

def plotRecyStats(data_HR, data_HR_DL, data_HRRecyDL,paramVal='None', plotDir="stat_analysis"):
    time = np.arange(data_HR.shape[0])

    std_HR = np.array( [np.std(x) for x in data_HR] )
    mean_HR = np.array( [np.mean(x) for x in data_HR] )

    std_HR_DL = np.array( [np.std(x) for x in data_HR_DL] )
    mean_HR_DL = np.array( [np.mean(x) for x in data_HR_DL] )

    std_HRRecyDL = np.array( [np.std(x) for x in data_HRRecyDL] )
    mean_HRRecyDL = np.array( [np.mean(x) for x in data_HRRecyDL] )

    plt.clf()
    plt.plot(time, mean_HR, '-k', label='True')
    plt.fill_between(time, mean_HR - std_HR, mean_HR + std_HR, color='k', alpha=0.2)

    plt.plot(time, mean_HR_DL, '-r', label='DL')
    plt.fill_between(time, mean_HR_DL - std_HR_DL, mean_HR_DL + std_HR_DL, color='r', alpha=0.2)

    plt.plot(time, mean_HRRecyDL, '-g', label='RecyDL')
    plt.fill_between(time, mean_HRRecyDL - std_HRRecyDL, mean_HRRecyDL + std_HRRecyDL, color='g', alpha=0.2)

    plt.xlabel('Time Iterations')
    plt.legend()
    plt.savefig(plotDir+"/timeDeviations_param"+str(paramVal)+".pdf" )
    print("Saved ",plotDir+"/timeDeviations_param"+str(paramVal)+".pdf"  )
    return; 
#plotRecyStats(data_HR, data_HR_DL, data_HRRecyDL,paramVal='None', plotDir="stat_analysis")

plotRecyStats(expected_HR, predicted_HR, predicted_HRRecy,paramVal='None', plotDir=".")


print("--EOF--")


