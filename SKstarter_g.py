# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

import os
import keras
import pydicom
from tqdm import tqdm_notebook
#from keras import backend as K
#from sklearn.model_selection import train_test_split
import gc
from skimage.transform import resize
import PIL
from tqdm import tqdm
import math
from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau,CSVLogger
#from keras.callbacks import EarlyStopping,LearningRateScheduler
from skimage import measure
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_similarity_score
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches


import json
from param_pixelmap import evaluate,generandom,geneaug,normAdapt,geneTable,spentTimeFunc
from param_pixelmap import normi
from param_pixelmap import maxadd, maxmult, maxrot, maxresize, maxshiftv, maxshifth,maxrotate,keepaenh
from param_pixelmap import load_weights_set
import cv2
import random
import time
import datetime
import argparse
from glob import glob
import numpy as np
import pickle
from cnn_pixelmap_model_V2 import get_model
import sys
from pyimagesearch.clr_callback import CyclicLR
from pyimagesearch.learningratefinder import LearningRateFinder

random.seed(42)

#path to ouput directory, it implies a set of charcteistics for training etc
#reportDir='r0nc'
#reportDir='r0ncD0'
reportDir='r0ncD0trial'
#reportDir='unetphoe'
#reportDir='unetreftrial'
reportDir='unetresnetrial'
#reportDir='unetref517'
#reportDir='merge'
#reportDir='unetresnet'
#reportDir='unetsimple'
#reportDir='r0nctrial'
#reportDir='unetresnet'
#reportDir='UResNet34'
reportDir='UResNet34trial'

#dirToConsider='r0ncD0_1024_modif' # for loading model
#dirToConsider='aws1024' # for loading model
#dirToConsider='wocosine' # for loading model
#dirToConsider='testLr' # for loading model
#dirToConsider='a16merge' # for loading model
dirToConsider='a' # for loading model





withScore=False # score for submission
withTrain=True # actual training
onePred=False
lookForLearningRate=False # actual training


withPlot= False# plot images during training
withPlot32= False# plot images and calculates min roi
withReport=False # calculate f score
withEval=False # evaluates as for scoring on validation data
calparam=False #calculate minroi

withScorePixel=False # calculate f score by pixel also
withRedoRoi=False # evaluates as for scoring on validation data
withMRoi=False
withmaskrcnn=False # use maskrcnn results for score True in merge
withrien=False # filter result

usePreviousWeight=False

zerocenter=True

#histAdapt='NAN' # use history adapt for pre -treatment
#histAdapt='clahe' # use history adapt for pre -treatment
histAdapt='histAdapt' # use history adapt for pre -treatment


withWeight=False # weighted class
weights=[0.1, 0.9]

augAskAsk=True #augmentation

writeModel=False

img_rows=512 #533
img_cols=512
nb_epoch=100
#nb_epoch=20


turnNumber=1
PercentVal=10 # and test
nsubSamples=100#in %
num_class=1
num_bit=1   
# work wit=h Kfolds
kfold=10
splitKfold=True

subreport=dirToConsider
minRoiForScore=0
thForScore=0.5
#threshold_best=0.5
prob_thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#prob_thresholds=[0.1,0.2]

#prob_thresholds=[0.5]

factorLR=0.5 # decay LR for each turn
minLR=1e-5 # min LR
MAX_LR = 1e-2
STEP_SIZE = 8
CLR_METHOD = "triangular2"
#######################################################################
if lookForLearningRate==True:
    withScore=False # score for submission
    withPlot= False# plot images and calculates min roi
    withPlot32= False# plot images and calculates min roi
    withReport=False # calculate f score
    withEval=False # evaluates as for scoring on validation data
    withScorePixel=False # calculate f score by pixel also
    withRedoRoi=False # evaluates as for scoring on validation data
    withMRoi=False
    onePred=False
    
if withTrain or calparam:
    withScore=False # score for submission
    withPlot= False# plot images and calculates min roi
    withPlot32= False# plot images and calculates min roi
    withReport=False # calculate f score
    withEval=False # evaluates as for scoring on validation data
    withScorePixel=False # calculate f score by pixel also
    withRedoRoi=False # evaluates as for scoring on validation data
    withMRoi=False
    onePred=False
    lookForLearningRate=False

    
if not withTrain:
    usePreviousWeight=True
    kfold=0


if not withTrain and withScore:
    withTrain=False # score for submission
    withPlot= False# plot images and calculates min roi
    withPlot32= False# plot images and calculates min roi
    modelName3=False # calculate f score
    withEval=False # evaluates as for scoring on validation data
    withScorePixel=False # calculate f score by pixel also
    withRedoRoi=False # evaluates as for scoring on validation data
    withMRoi=False


minRoiToConsider={}
repDir={}
batchSize={}
d0Dict={}
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)

start = time.time()
tn = datetime.datetime.now()

ap = argparse.ArgumentParser()
ap.add_argument("-dir", "--dir", 
    default=reportDir,
	help="path to output directory top")

args = vars(ap.parse_args())
reportDir = args["dir"]


if reportDir=='unetphoe':
    modelName='unetphoe'
    mergeDir=False
    learning_rate=1e-3
    d0=0.5
    batchSize[modelName]=16 # 3 for create network 512 1 dor 1024

if reportDir=='unetresnet':
    modelName='unetresnet'
    mergeDir=False
    learning_rate=minLR
    d0=0.5
    batchSize[modelName]=7# 3 for create network 512 1 dor 1024
    if lookForLearningRate:
            kfold=0
            nb_epoch=10

if reportDir=='UResNet34':
    modelName='UResNet34'
    mergeDir=False
    learning_rate=minLR
    d0=0.5
    nb_epoch=100
    batchSize[modelName]=16# 16 OK
    if lookForLearningRate:
            kfold=0
            nb_epoch=10

if reportDir=='unetsimple':
    modelName='unetsimple'
    mergeDir=False
    learning_rate=1e-3
    d0=0.5
    batchSize[modelName]=2 # 3 for create network 512 1 dor 1024


if reportDir=='unetresnetrial':
    modelName='unetresnet'
    random.seed(42)
    kfold=0
    splitKfold= False

    d0=0.5
    mergeDir=False
    learning_rate=minLR
    img_rows=256
    img_cols=256
    batchSize[modelName]=30  # 10 for create network 512
    nb_epoch=1
    turnNumber=1
    PercentVal=10 # and test
    nsubSamples=15#in %
    minRoiToConsider[img_rows]=80# start 1300.for 512
    prob_thresholds=[0.3,0.5,0.6]
    thForScore=0.5


if reportDir=='UResNet34trial':
    modelName='UResNet34'
    random.seed(42)
    kfold=0
    splitKfold= False

    d0=0.5
    mergeDir=False
    learning_rate=minLR
    img_rows=256
    img_cols=256
    batchSize[modelName]=30  # 10 for create network 512
    nb_epoch=1
    turnNumber=1
    PercentVal=10 # and test
    nsubSamples=15#in %
    minRoiToConsider[img_rows]=80# start 1300.for 512
    prob_thresholds=[0.3,0.5,0.6]
    thForScore=0.5


    
if reportDir=='unetref':
    modelName='unet'
    mergeDir=False
    d0=0.04 #0.04
    learning_rate=1e-3
    batchSize[modelName]=6  # 6 for unet 512 4 for 1024


if reportDir=='unetreftrial':
    modelName='unet'
    random.seed(42)
    kfold=10

    d0=0.04 #=0.04
    mergeDir=False
    learning_rate=1e-4
    img_rows=256
    img_cols=256
    batchSize[modelName]=10  # for create network 512


if reportDir=='r0ncD0':
    modelName='downsample_resblock'
    mergeDir=False
    learning_rate=minLR
#    d0=0.
    d0=0.5
#    learning_rate=1e-3
    batchSize[modelName]=8 # 8 for create network 512, 4 for 1024


      
if reportDir=='r0ncD0trial':
    random.seed(42)
    modelName='downsample_resblock'
    mergeDir=False
    kfold=3
    d0=0.5
    learning_rate=minLR
    img_rows=128
    img_cols=128
    nsubSamples=10#in %
    nb_epoch=20
    turnNumber=2
    batchSize[modelName]=10 # for create network 512
 
#    prob_thresholds=[0.5]
      
if reportDir=='merge':
    mergeDir=True
    random.seed(42)
    d0=d0Dict
    modelName='unetresnet'
    modelName2='create_network'
    modelName3='unet'
    withmaskrcnn=False
    pickleMasrcnn=os.path.join('../aws/maskrcnn50','predmaskrcnnBox.pkl')
    d0Dict['create_network']=0.5
    d0Dict['unetresnet']=0.5
    d0Dict['unet']=0.04
    repDir['create_network']='../r0ncD0/'+dirToConsider
    repDir['unet']          ='../unetref/'+dirToConsider
    repDir['unetresnet']    ='../unetresnet/'+dirToConsider
    learning_rate=0.001
    batchSize[modelName]=3 # for create network 512
    batchSize[modelName2]=12 # for create network 512
    batchSize[modelName3]=8 # for create network 512

#minRoiToConsider[128]=100 # start 1300.for 512
#minRoiToConsider[256]=325 # start 1300.for 512
#minRoiToConsider[512]=1300 # start 1300.for 512
#minRoiToConsider[1024]=5200 # start 1300.for 512
##r1024=1024./img_rows
##minRoiToConsider[1024]=r1024*r1024*minRoiToConsider[img_rows]
#minRoiVal=minRoiToConsider[img_rows]

#miRoiList= [0,minRoiVal/4.0,minRoiVal/2.0, 3.*minRoiVal/4,minRoiVal]
miRoiList=[0]

reportDir=os.path.join(cwdtop,reportDir,subreport)

if not os.path.exists(reportDir) :
    os.makedirs(reportDir)

print ('work directory: ',reportDir)

print ('start: '+ str(tn.month)+'month '+str(tn.day)+'day at: '+str(tn.hour)+'hours '+
       str(tn.minute)+'minutes '+str(tn.second)+'seconds')


print('image size:',img_rows,img_cols)
todaytrsh='m'+str(tn.month)+'_d'+str(tn.day)+'_y'+str(tn.year)+'_'+str(tn.hour)+'h_'+str(tn.minute)+'m_trsh.txt'
todayracine='m'+str(tn.month)+'_d'+str(tn.day)+'_y'+str(tn.year)+'_'+str(tn.hour)+'h_'+str(tn.minute)
todaytrshFile=os.path.join(reportDir,todaytrsh)    
f=open(todaytrshFile,'w')
#print str(tn.month)+'_'+str(tn.day)+'_'+str(tn.year)+'_'+str(tn.hour)+'_'+str(tn.minute)
f.write('start: '+ str(tn.month)+'month '+str(tn.day)+'day at: '+str(tn.hour)+'hours '+
        str(tn.minute)+'minutes '+str(tn.second)+'seconds\n')
f.write('reportDir: '+str(reportDir)+'\n')
f.write('sub dir to load model: '+str(dirToConsider)+'\n')
f.write('modelName: '+str(modelName)+'\n')
f.write('d0: '+str(d0)+'\n')
f.write('img_rows: '+str(img_rows)+'\n')
f.write('img_cols: '+str(img_cols)+'\n')
f.write('batchSize: '+str(batchSize[modelName])+'\n')
f.write('nb_epoch: '+str(nb_epoch)+'\n')
f.write('turnNumber: '+str(turnNumber)+'\n')
f.write('nsubSamples: '+str(nsubSamples)+'\n')
f.write('PercentVal: '+str(PercentVal)+'\n')
f.write('keepaenh: '+str(keepaenh)+'\n')
f.write('maxmult: '+str(maxmult)+'\n')
f.write('maxadd: '+str(maxadd)+'\n')
f.write('maxrot: '+str(maxrot)+'\n')
f.write('maxresize: '+str(maxresize)+'\n')
f.write('maxshiftv: '+str(maxshiftv)+'\n')
f.write('maxshifth: '+str(maxshifth)+'\n')
f.write('maxrotate: '+str(maxrotate)+'\n')
f.write('learning_rate: '+str(learning_rate)+'\n')
f.write('factorLR: '+str(factorLR)+'\n')
f.write('minLR: '+str(minLR)+'\n')
f.write('MAX_LR: '+str(MAX_LR)+'\n')
f.write('STEP_SIZE: '+str(STEP_SIZE)+'\n')
f.write('CLR_METHOD: '+str(CLR_METHOD)+'\n')

f.write('number of kfold: '+str(kfold)+'\n')
f.write('split kfold: '+str(splitKfold)+'\n')

f.write('zerocenter: '+str(zerocenter)+'\n')
#f.write('minRoiToConsider: '+str(minRoiToConsider[img_rows])+'\n')
f.write('thForScore: '+str(thForScore)+'\n')
f.write('use Histogram adapt: '+str(histAdapt)+'\n')
f.write('use data augmentation: '+str(augAskAsk)+'\n')
f.write('use PreviousWeight: '+str(usePreviousWeight)+'\n')
f.write('withWeight for classes: '+str(withWeight)+'\n')
if withWeight:
    f.write('weights for classes: '+str(weights)+'\n')


try:
    f.write('use maskrcnn for score: '+str(withmaskrcnn)+' : '+str(pickleMasrcnn)+'\n')
except:
        pass
f.write('filter input: '+str(withrien)+'\n')

f.write('-------------------------------------\n')
f.close()

with open('/home/sylvain/Documents/trial/siim/python/settings.json') as json_data_file:
        json_data=json.load(json_data_file)

RAW_TRAIN_LABELS=json_data['RAW_TRAIN_LABELS']
TRAIN_DCM_DIR=json_data["TRAIN_DCM_DIR"]
TRAIN_DCM_SUB=json_data["TRAIN_DCM_SUB"]
TEST_DCM_DIR=json_data["TEST_DCM_DIR"]
#print(TRAIN_DCM_DIR)

folderTrain = TRAIN_DCM_DIR

today = str('m'+str(tn.month)+'_d'+str(tn.day)+'_y'+str(tn.year)+'_'+str(tn.hour)+'h_'+str(tn.minute)+'m')

#            print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
def getfilenames(): 

#    tr = pd.read_csv(RAW_TRAIN_LABELS)
#getting path of all the train and test images
    file_train=sorted(glob(TRAIN_DCM_DIR+'*/*/*/*.dcm'))
    test_fns=sorted(glob(TEST_DCM_DIR+'*/*/*/*.dcm'))
    
    #train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
    #test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))
    
    print(len(file_train))
    print(len(test_fns))
    pneumothorax=[]
#df = pd.read_csv(RAW_TRAIN_LABELS, header=None, index_col=0)
    df_full = pd.read_csv(RAW_TRAIN_LABELS, index_col='ImageId')
    
    for n, _id in tqdm_notebook(enumerate(file_train), total=len(file_train)):
        try:
                if not '-1'  in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
        #            Y_train[n] = np.zeros((1024, 1024, 1))
                    pneumothorax.append(_id.split('/')[-1])
        except KeyError:
            pass

    print  ( 'total number of images:',len(file_train))
    f=open(todaytrshFile,'a')
    f.write('total number of images: ' +str(len(file_train))+'\n')
    file_with_pneumo = [name for name in file_train if name.split('/')[-1]  in pneumothorax]

    print ('number of images with pnemonia:', len(file_with_pneumo))
    f.write('number of images with pnemonia: ' +str(len(file_with_pneumo))+'\n')
    
    file_without_pneumo = [name for name in file_train if name.split('/')[-1] not in pneumothorax]

    print ('number of images with no pnemonia:', len(file_without_pneumo))
    f.write('number of images with no pnemonia: ' +str(len(file_without_pneumo))+'\n')
    random.shuffle(file_without_pneumo)
    file_without_pneumo_reduc=file_without_pneumo[0:len(file_with_pneumo)//10]
#    filenames3=filenames2
    filenames=file_with_pneumo+file_without_pneumo_reduc
    random.shuffle(filenames)
    
    print ('final number of images:',len(filenames))
    f.write('final number of images: ' +str(len(filenames))+'\n')
  
    print ('subsample',nsubSamples,'%')    
    filenames=filenames[:int(len(filenames)*nsubSamples/100.)]
    print ('final number of subsampled images:',len(filenames))
    f.write('final number of subsampled images: ' +str(len(filenames))+'\n')
    f.close()
  
        # split into train and validation filenames
    if kfold ==0:
    #n_valid_samples = 2560
        print ('percentage for validation',PercentVal,'%')
        n_valid_samples = int(len(filenames)*PercentVal/100)
        #print  'number valid',n_valid_samples
        train_filenames = filenames[n_valid_samples:]
        valid_filenames = filenames[:n_valid_samples]
        print('n train samples', len(train_filenames))
        print('n valid samples', len(valid_filenames))
    #    n_train_samples = len(filenames) - n_valid_samples
        
    # 
    
        f=open(todaytrshFile,'a')
        f.write('number train samples: ' +str(len(train_filenames))+'\n')
        f.write('number valid samples: ' +str(len(valid_filenames))+'\n')
        
    #    print ('min area: ',minarea,' square min area: ',math.sqrt(minarea),' maxarea:',maxarea,' square max area: ',math.sqrt(maxarea),\
    #           ' 1024**2: ',1024*1024,' max scale: ', maxscale,' minscale: ',minscale)
    #    f.write('min area: '+str(minarea)+' square min area: '+str(math.sqrt(minarea))+' maxarea:'+str(maxarea)+' square max area: '+str(math.sqrt(maxarea))+\
    #           ' 1024**2: '+str(1024*1024)+' max scale: '+ str(maxscale)+' minscale: '+str(minscale)+'\n')
        f.close()
    else:
        kf = KFold(n_splits=kfold)

        train_filenames={}
        valid_filenames={}
        i=-1
        for train_index, test_index in kf.split(filenames):
            i+=1
            train_filenames[i]=[]
            valid_filenames[i]=[]
#            print(train_index,test_index)
            for k in train_index:
                train_filenames[i].append(filenames[k])
            for k in test_index:
                valid_filenames[i].append(filenames[k])
#            valid_filenames[i]=filenames[test_index]
        print('number of folders',kf.get_n_splits(filenames))
        print('n train samples', len(train_filenames[0]))
        print('n valid samples', len(valid_filenames[0]))
            #    n_train_samples = len(filenames) - n_valid_samples
                
            # 
            
        f=open(todaytrshFile,'a')
        f.write(('number of folders'+ str(kf.get_n_splits(filenames)))+'\n')
        f.write('number train samples: ' +str(len(train_filenames[0]))+'\n')
        f.write('number valid samples: ' +str(len(valid_filenames[0]))+'\n')
        f.close()

            
    return train_filenames,valid_filenames,pneumothorax,df_full

            

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;
    img8=img.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
#    print(img.shape,img.min(),img.max())
    _, maskd = cv2.threshold(img8, 200, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(maskd, kernel, iterations=2)
    _, img8 = cv2.threshold(erode, 200, 255, cv2.THRESH_BINARY)
    for x in range(width):
        for y in range(height):
            currentColor = img8[x][y]
            if currentColor != lastColor: 
                if currentColor >= 127:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    
    mask= np.zeros(width* height,np.uint8)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)
    array = np.asarray([int(x) for x in rle.split()])
    
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
#    mask=mask.T.copy()
    mask=mask.reshape(width, height)
    _, maskd = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(maskd, kernel, iterations=2)
    _, dilation = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)
    return dilation

def rle2mask1(rle, width, height):
    
    mask= np.zeros(width* height,np.uint8)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)
    array = np.asarray([int(x) for x in rle.split()])
    
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
    mask=mask.reshape(width, height)
#    mask=mask.T.copy()
    _, maskd = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    erode = cv2.erode(maskd, kernel, iterations=10)
    _, erode = cv2.threshold(erode, 200, 255, cv2.THRESH_BINARY)
#    print(np.unique(dilation))

    return erode


def rle2mask0(rle, width, height):
    
    mask= np.zeros(width* height,np.uint8)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)
    array = np.asarray([int(x) for x in rle.split()])
    
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
#    mask=mask.T.copy()
   
#    print(np.unique(dilation))

    return mask.reshape(width, height)

def maskfromrle(df_full,f,img_rows,img_cols,rle2m):
     if type(df_full.loc[f[:-4],' EncodedPixels']) == str:
            msk = rle2m(df_full.loc[f[:-4],' EncodedPixels'], 1024, 1024)
     else:
        msk= np.zeros((1024, 1024),np.uint16)
        for x in df_full.loc[f[:-4],' EncodedPixels']:                      
            msk =  np.clip(msk + rle2m(x, 1024, 1024),0,255)
     
    #print('mask',mask.min(),mask.max())
     msk=(cv2.resize(msk,(img_rows,img_cols),interpolation=cv2.INTER_NEAREST))
     msk=msk.astype(np.uint8)
     return msk.T

def test():
        
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
    vf=['1.2.276.0.7230010.3.1.4.8323329.5577.1517875188.867087.dcm','1.2.276.0.7230010.3.1.4.8323329.10012.1517875220.965942.dcm']
#    vf=['1.2.276.0.7230010.3.1.4.8323329.10012.1517875220.965942.dcm']
    
    #    for i in range(0,10):'86459db9-dc9d-42f5-818d-9525684d5406.dcm', '74504e90-1547-40f1-9059-ade3f2576278.dcm'
    for filename in vf:
    
    #        filename='b6862fc0-31f9-4091-b8b1-256192168a0f.dcm'
    #        filename=valid_filenames[i]
        print (filename)
    #    mask=np.zeros((img_rows,img_cols,3),np.uint8)
    #        imagePath=os.path.join(folderTrain,filename)
        filename=os.path.join('../input/refDicom',filename)
    #for i in range (9,10):
    #    filename=train_filenames[i]
    #    
    #    mask=np.zeros((img_rows,img_cols,3),np.uint8)
        
        imgt,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(filename,img_rows,img_cols,dirWriteBmp='',
                                                writeBmp=False,resize=True,histAdapt=histAdapt)
        img=normAdapt(imgt,MIN_BOUND,MAX_BOUND,zerocenter)
        f=filename.split('/')[-1]
        msk0=maskfromrle(df_full,f,img_rows,img_cols,rle2mask0)
        msk=maskfromrle(df_full,f,img_rows,img_cols,rle2mask)
        msk1=maskfromrle(df_full,f,img_rows,img_cols,rle2mask1)
        
        print('msk0',msk0.min(),msk0.max())
        print('msk',msk.min(),msk.max())
        print('msk1',msk1.min(),msk1.max())
    
        
    
        #        plt.imshow(normi(imgt),cmap = 'gray')
        #        plt.show()
        #        plt.imshow(normi(msk))
        #        plt.show()
    
        mkadd=cv2.addWeighted(normi(img),0.5,normi(msk),0.5,0)
        plt.imshow(normi(mkadd))
        plt.show()
        mkadd=cv2.addWeighted(normi(img),0.5,normi(msk1),0.5,0)
        plt.imshow(normi(mkadd))
        plt.show()
        mkadd=cv2.addWeighted(normi(img),0.5,normi(msk0),0.5,0)
        plt.imshow(normi(mkadd))
        plt.show()
        msk01=msk-msk0
        print('msk01',msk01.min(),msk01.max())
        plt.imshow(normi(msk01))
        plt.show()
        
#
#test()
#ooo
##
##
#
#
#tr = pd.read_csv(RAW_TRAIN_LABELS)
#print(tr[' EncodedPixels'][5])
#ooo
#mask=rle2mask(tr[' EncodedPixels'][5],1024,1024)
#print('mask',mask.min(),mask.max())
#plt.imshow(mask)
#plt.show()
#cv2.imwrite('a.bmp',mask)
#_, maskd = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
#print('maskd',maskd.min(),maskd.max())
#kernel = np.ones((3,3), np.uint8)
#dilation = cv2.dilate(maskd, kernel, iterations=2)
#print('dilation',dilation.min(),dilation.max())
#
#cv2.imwrite('b.bmp',dilation)
#maskdif=dilation-mask
#plt.imshow(dilation)
#plt.show()
#plt.imshow(maskdif)
#cv2.imwrite('c.bmp',maskdif)
#print('maskdif',maskdif.min(),maskdif.max())
#plt.show()
#ooo

#ooo
def get_mask(encode,width,height):
    if encode == [] or encode == ' -1':
        return rle2mask(' -1',width,height)
    else:
        return rle2mask(encode[0],width,height)       



def image_n_encode(train_images_names,encode_df):
    train_imgs = [] 
    train_encode = []
    c = 0
    for f in tqdm_notebook(train_images_names):
        if c >= 2000:
            break
        try:
            img = pydicom.read_file(f).pixel_array
            c += 1
            encode = list(encode_df.loc[encode_df['ImageId'] == '.'.join(f.split('/')[-1].split('.')[:-1]),
                               ' EncodedPixels'].values)
            
            encode = get_mask(encode,img.shape[1],img.shape[0])
            encode = resize(encode,(img_rows,img_cols))
            train_encode.append(encode)
            img = resize(img,(img_rows,img_cols))
            train_imgs.append(img)
        except pydicom.errors.InvalidDicomError:
            print('come here')
        
    return train_imgs,train_encode
        
#
def cosine_annealing(x):
#    lr = learning_rate
#    epochs = nb_epoch
    return max(learning_rate*(np.cos(np.pi*x/nb_epoch)+1.)/2,minLR)
learning_rates = keras.callbacks.LearningRateScheduler(cosine_annealing)




class generator(keras.utils.Sequence):
    
    def __init__(self, filenames, df_full, pneumothorax=None, batch_size=32, 
                 image_size=(256,256), shuffle=True, augment=False, predict=False):
        self.filenames = filenames
        self.df_full = df_full
        self.pneumothorax = pneumothorax
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()


    def __load__(self, filename):
        # load dicom file as numpy array
        img,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(filename,self.image_size[0],self.image_size[1],dirWriteBmp='',
                                            writeBmp=False,resize=True,histAdapt=histAdapt)
#        img = pydicom.read_file(filename).pixel_array
#        img = resize(img,self.image_size)

        msk = np.zeros(self.image_size,np.uint8)
        # get filename without extension
        filename = filename.split('/')[-1]
        # if image contains pneumonia
        if filename in self.pneumothorax:
            msk=maskfromrle(self.df_full,filename,self.image_size[0],self.image_size[1],rle2mask)
#            if type(self.df_full.loc[filename[:-4],' EncodedPixels']) == str:
#                msk = rle2mask(self.df_full.loc[filename[:-4],' EncodedPixels'], 1024, 1024)
##                print('good1',msk.max())
#                    
#            else:
#                    msk= np.zeros((1024, 1024),np.uint16)
#                    for x in self.df_full.loc[filename[:-4],' EncodedPixels']:                      
#                        msk =  np.clip(msk + rle2mask(x, 1024, 1024),0,255)
#                        msk=msk.astype(np.uint8)
##                    print('good2',msk.max())
##                    plt.imshow(msk)
##                    plt.show()
##       
#        msk=cv2.resize(msk,(self.image_size[0],self.image_size[1]),interpolation=cv2.INTER_NEAREST)
#        msk=msk.T
#        print('after resize',msk.max())
        if self.augment:
            addint,multint,rotimg,resiz,shiftv,shifth,rotateint=generandom(maxadd,
                                    maxmult,maxrot,maxresize,maxshiftv,maxshifth,maxrotate,keepaenh)
            img=geneaug(img,addint,multint,rotimg,resiz,shiftv,shifth,rotateint,False,img.min(),img.max())
            msk=geneaug(msk,  0,    0,     rotimg,resiz,shiftv,shifth,rotateint,True,msk.min(),msk.max())
#            img=datagen.apply_transform(img)
#            msk=datagen.apply_transform(msk)

        img=normAdapt(img,MIN_BOUND,MAX_BOUND,zerocenter)
        
        # add trailing channel dimension
        
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)/255.
#        print(img.shape)
#        print(msk.shape)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
#        imagePath=os.path.join(self.folder, filename)
        img,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(filename,self.image_size[0],self.image_size[1],dirWriteBmp='',
                                            writeBmp=False,resize=True,histAdapt=histAdapt)
        img=normAdapt(img,MIN_BOUND,MAX_BOUND,zerocenter)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
#            print(np.unique(msks))
            if (len(np.unique(msks)))>2:
                print('error more than 2 categories')
                sys.exit()

#            print('img',imgs.shape,imgs.min(),imgs.max())
#            print('mask',msks.shape,msks.min(),msks.max())
#            imgs,msks=formatData(imgs,msks,num_class)

            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
# learning rate schedule
def step_decay(epoch):
	drop = 0.5
	epochs_drop = 20.0
	lrate = learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate







def trainAct(tf,vf,kf,df_full,pneumothorax,model,lastepoch,len_train):
    global learning_rate
    startlocal = time.time()

    BATCH_SIZE = batchSize[modelName]

#    NUM_EPOCHS = 96
    
    print("[INFO] using '{}' method".format(CLR_METHOD))
    clr = CyclicLR(
    	mode=CLR_METHOD,
    	base_lr=minLR,
    	max_lr=MAX_LR,
    	step_size= STEP_SIZE * (len_train// BATCH_SIZE))
    
    #early stopping: 1.5 full set without increase: step-size*3 = 24
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=STEP_SIZE*3, verbose=1,min_delta=0.005,mode='min')  
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                                              patience=5, min_lr=1e-7,
#                                              verbose=1,mode='auto')#init 5
    if kfold>0 and splitKfold:
        lrdir=os.path.join(reportDir,str(kf))
        if not os.path.exists(lrdir) :
            os.makedirs(lrdir)
    else:
            lrdir=reportDir
#    print(lrdir)
    
#    checkpoint_path = os.path.join(lrdir,str(today)+'-{epoch:02d}-{val_dice_coef:.4f}.hdf5')
    checkpoint_path = os.path.join(lrdir,str(today)+'-{epoch:02d}-{val_dice_coef:.4f}.hdf5')

#    checkpoint_path2 = os.path.join(lrdir,str(today)+'_dice-coef_{epoch:02d}-{val_acc:.2f}.hdf5')
    fileResult=os.path.join(lrdir,str(today)+'_e.csv')



    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                       save_weights_only=True,  mode='min')

#    model_checkpoint2 = keras.callbacks.ModelCheckpoint(checkpoint_path2, verbose=1,
#                                monitor='val_dice_coef', save_best_only=True,save_weights_only=True,mode='max') 
    csv_logger = CSVLogger(fileResult,append=True)

    batch_size=batchSize[modelName]

#    lastepoch=0


    print ('split number: ',kf+1, ' on: ', max(1,kfold),'nb epoch: ',nb_epoch,' learning rate: ',learning_rate)
    f=open(todaytrshFile,'a')
    f.write('-----------------\n')
    f.write('split number: '+str(kf+1)+' on: '+str(max(1,kfold))+' nb epoch:' +
            str(nb_epoch)+' learning rate: '+str(learning_rate)+'\n')
    f.close()
#        train_filenames,valid_filenames,maskRoi=getfilenames()
    nb_epoch_i_p=lastepoch+nb_epoch

    # create train and validation generators
    train_gen = generator(tf,df_full,pneumothorax, 
                          batch_size=batch_size, image_size=(img_rows,img_cols), 
                          shuffle=True,  augment=augAskAsk, predict=False)
#        class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(train_gen),
#                                                 train_gen)
    valid_gen = generator(vf,df_full, pneumothorax, 
                          batch_size=batch_size, image_size=(img_rows,img_cols), 
                          shuffle=False, augment=False, predict=False)

    history = model.fit_generator(train_gen, validation_data=valid_gen, 

                               epochs=nb_epoch_i_p,
                               initial_epoch=lastepoch,
  
#                                  callbacks=[model_checkpoint,learning_rates,csv_logger,early_stopping],
#                                  callbacks=[model_checkpoint,csv_logger,early_stopping,
#                                             reduce_lr,model_checkpoint2],
#                              callbacks=[model_checkpoint,csv_logger,early_stopping,
#                                         reduce_lr],
                              callbacks=[model_checkpoint,csv_logger,early_stopping, clr],
#                              callbacks=[csv_logger,early_stopping, clr],


                               steps_per_epoch=len(tf)//batchSize[modelName],
#                               callbacks=[model_checkpoint,clr,csv_logger],    
                              workers=4, use_multiprocessing=True)
    
    
    
    
    # plot the learning rate history
#    N = np.arange(0, len(clr.history["lr"]))
#    plt.figure()
#    plt.plot(N, clr.history["lr"])
#    plt.title("Cyclical Learning Rate (CLR)")
#    plt.xlabel("Training Iterations")
#    plt.ylabel("Learning Rate")
#    plt.show()
    nb_epochs=len(history.history['loss'])
    if not splitKfold:
        lastepoch=int((kf+1)*nb_epochs)
    else:
        lastepoch=0
    print (history.history.keys())


    f=open(todaytrshFile,'a')
    print('actual number of epochs',nb_epochs)
    f.write('actual number of epochs: '+str(nb_epochs)+'\n')

    spenttime=spentTimeFunc(time.time() - startlocal)
    print ('turn training time spent:' ,spenttime)  
    f.write('turn training time spent: '+str(spenttime)+'\n')
    f.close()
    
    if withPlot:
        todayPlotIou='r'+'_'+str(tn.month)+'_'+str(tn.day)+'_'+str(tn.year)+'_'+str(tn.hour)+'_'+str(tn.minute)+'plot.png'
        plotFileIou=os.path.join(reportDir,todayPlotIou)    
        
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.plot(history.epoch, history.history["loss"], label="Train loss")
        plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
        plt.legend()
        
        plt.subplot(132)
        plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
        plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
        plt.legend()
        
        plt.subplot(133)
        try:
           
            plt.plot(history.epoch, history.history["mean_iou"], label="Train iou")
            plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
        except:
            plt.plot(history.epoch, history.history["dice_coef"], label="dice_coef")
            plt.plot(history.epoch, history.history["val_dice_coef"], label="val_dice_coef")
            
        plt.legend()
        plt.savefig(plotFileIou)
        plt.show()

    
#    images_reshape=np.array(images).reshape(-1,img_rows,img_cols,1)
#    mask_e_reshape=np.array(mask_e).reshape(-1,img_rows,img_cols,1)
##    
##    print(images_reshape.shape,images_reshape.min(),images_reshape.max())
##    print(mask_e_reshape.shape,mask_e_reshape.min(),mask_e_reshape.max())
#    model.fit(images_reshape,mask_e_reshape,validation_split = 0.1,epochs = 1,batch_size = 16)
    gc.collect()
    return lastepoch
    

def withTrainf():
    global learning_rate
    print('start training on: ',max(kfold,1), ' folders')
    lastepoch=0
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
    if kfold>0:
        len_train=len(train_filenames[0])
    else:
        len_train=len(train_filenames)
    if kfold>0:
#        print(len(train_filenames[0]))
#        print(len(valid_filenames[0]))
        if batchSize[modelName] >len(valid_filenames[0]):
            print('error batch size > len valid')
            print(len(valid_filenames[0]),batchSize[modelName])
            sys.exit()

    else:
#         print(len(train_filenames))
#         print(len(valid_filenames))
         if batchSize[modelName] >len(valid_filenames):
            print('error batch size > len valid')
            print(len(valid_filenames),batchSize[modelName])
            sys.exit()

    if kfold>0:
        for i in range(kfold):
            print('start with folder:',i, 'on: ',kfold-1)
            if splitKfold:
                if mergeDir:
                        model,model2,model3=loadModelGlobal(i)
                else:
                        model=loadModelGlobal(i)
            else:
                if i==0:
                    if mergeDir:
                            model,model2,model3=loadModelGlobal(i)
                    else:
                            model=loadModelGlobal(i)
                            
            lastepoch=trainAct(train_filenames[i],valid_filenames[i],i,df_full,
                               pneumothorax,model,lastepoch,len_train)
            
    else:
            if mergeDir:
                    model,model2,model3=loadModelGlobal(0)
            else:
                    model=loadModelGlobal(0)
            lastepoch=trainAct(train_filenames,valid_filenames,0,df_full,
                               pneumothorax,model,lastepoch,len_train)
    
    
#    model.fit(images_reshape,mask_e_reshape,validation_split = 0.1,epochs = 1,batch_size = 16)
    del train_filenames,valid_filenames,df_full,model
    f=open(todaytrshFile,'a')
    spenttime=spentTimeFunc(time.time() - start)
    print ('total training time spent :' ,spenttime  )
    f.write('total time spent: '+str(spenttime)+'\n')
    f.close()
    gc.collect()



def trainActLfLR(tf,vf,kf,df_full,pneumothorax,model,lastepoch,len_train):
    batch_size=batchSize[modelName]

    train_gen = generator(tf,df_full,pneumothorax, 
                          batch_size=batch_size, image_size=(img_rows,img_cols), 
                          shuffle=True,  augment=augAskAsk, predict=False)
    print("[INFO] finding learning rate...")
    
    lrf = LearningRateFinder(model)
    lrf.find(train_gen,
#		aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
		1e-10, 1e+1,
        epochs=nb_epoch,
        stepsPerEpoch=len(tf)//batch_size,

#		stepsPerEpoch=np.ceil((len(trainX) / float(config.BATCH_SIZE))),
		batchSize=batch_size)
    
    
    lrf.plot_loss()
    
    
    


def lookForLearningRatef():
    lastepoch=0
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
    len_train=len(train_filenames[0])
    
    
    if kfold>0:
        for i in range(kfold):
            print('start with folder:',i, 'on: ',kfold-1)
            if splitKfold:
                if mergeDir:
                        model,model2,model3=loadModelGlobal(i)
                else:
                        model=loadModelGlobal(i)
            else:
                if i==0:
                    if mergeDir:
                            model,model2,model3=loadModelGlobal(i)
                    else:
                            model=loadModelGlobal(i)
                            
            lastepoch=trainActLfLR(train_filenames[i],valid_filenames[i],i,df_full,
                               pneumothorax,model,lastepoch,len_train)
            
    else:
            if mergeDir:
                    model,model2,model3=loadModelGlobal(0)
            else:
                    model=loadModelGlobal(0)
            lastepoch=trainActLfLR(train_filenames,valid_filenames,0,df_full,
                               pneumothorax,model,lastepoch,len_train)
    
    
#    model.fit(images_reshape,mask_e_reshape,validation_split = 0.1,epochs = 1,batch_size = 16)
    del train_filenames,valid_filenames,df_full,model
    f=open(todaytrshFile,'a')
    spenttime=spentTimeFunc(time.time() - start)
    print ('total training time spent :' ,spenttime  )
    f.write('total time spent: '+str(spenttime)+'\n')
    f.close()
    gc.collect()





def withReportf():
    print ('----------------------------------------')
    random.seed(42)
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
    print ('calculate report f-score on ',len(valid_filenames),' validation data')
    f=open(todaytrshFile,'a')
    f.write('calculate report f-score on: '+str(len(valid_filenames))+' validation data\n')
    f.write('----------------------------------------------------------------\n')
    f.close()

    lvfn=len(valid_filenames)  
    try:
        basize= 20*min(batchSize[modelName],batchSize[modelName2],batchSize[modelName3])
    except:
        basize= 20*batchSize[modelName]
    valid_gen = generator(valid_filenames,df_full, pneumothorax, 
                              batch_size=basize, image_size=(img_rows,img_cols), 
                              shuffle=False, augment=False, predict=False)
   
    roiGoodacc =0
    roiGoodfs=0
    roiGoodrecall=0
    
    thGoodacc=1
    thGoodfs=1
    thGoodrecall=1
    
    accGood=0
    fsGood=0
    recallGood=0
    
#    bA=False
    indb=-1
    for imgs, msks in valid_gen:
        indb+=1

        preds = model.predict(imgs,batch_size=batchSize[modelName],verbose=1)
        if mergeDir:          
            preds2 = model2.predict(imgs,batch_size=batchSize[modelName2],verbose=1)     
            preds3 = model3.predict(imgs,batch_size=batchSize[modelName3],verbose=1)    
            preds=(preds+preds2+preds3)/3
            preds2=[]
            preds3=[]

        if indb==0:
            predTot=preds.copy()
            maskTot=msks.copy()
        else:
            predTot=np.concatenate((predTot,preds),axis=0)
            maskTot=np.concatenate((maskTot,msks),axis=0)
        if maskTot.shape[0]>=lvfn:
            predTot=predTot[:lvfn]
            maskTot=maskTot[:lvfn]
            break

    preds=[]
    kernel=np.ones((3,3),np.uint8)  
      
    taby=np.zeros((len(miRoiList),len(prob_thresholds)),np.float)
    tabx=np.zeros((len(miRoiList),len(prob_thresholds)),np.float)
    tabz=np.zeros((len(miRoiList),len(prob_thresholds)),np.float)
    iminroi=-1
    for minRoif in miRoiList:
        iminroi+=1
        ith=-1
#        print minRoif
        for th in prob_thresholds:
            ith+=1
            taby[iminroi][ith]=th
            tabx[iminroi][ith]=minRoif
#            print th
            y_true=np.zeros(lvfn,np.uint8)
            y_pred=np.zeros(lvfn,np.uint8)
            for ind,( msk, predorig) in enumerate(zip(maskTot, predTot)):
                    # threshold true mask
                mskt = msk[:, :, 0] > 0.5

                if mskt.max()==True:
                    y_true[ind]=1
                pred=predorig.copy()
                np.putmask(pred,predorig > th,255)
                np.putmask(pred,predorig <= th,0)
                pred = pred.reshape(img_rows,img_cols)
                _, maskd = cv2.threshold(pred, 200, 255, cv2.THRESH_BINARY)
                kernel1 = np.ones((3,3), np.uint8)
                erode = cv2.erode(maskd, kernel1, iterations=2)
                _, pred = cv2.threshold(erode, 200, 255, cv2.THRESH_BINARY)
#                if mskt.max()==True:
#                    plt.imshow(normi(pred))
#                    plt.show()
#                    plt.imshow(normi(mskt))
#                    plt.show()
#                    ooo
#                pred2=pred.copy()
                pred = cv2.dilate(pred, kernel,iterations=3) 
                pred = cv2.erode(pred, kernel,iterations=3)
                pred = cv2.dilate(pred, kernel,iterations=3) 
                pred = cv2.erode(pred, kernel,iterations=3) 
#                if mskt.max()==True:
#                    plt.imshow(normi(pred))
#                    plt.show()
#                    plt.imshow(normi(pred2-pred))
#                    plt.show()
#                    ooo
                
                predm = measure.label(pred)
                
                

                for region in measure.regionprops(predm):
                    # retrieve x, y, height and width
                    y, x, y2, x2 = region.bbox
                    height = y2 - y
                    width = x2 - x
                    if height*width>minRoif:
                        y_pred[ind]=1
                        break
                    if height*width<0:
                        print ('error negative')
                        sys.exit()
            print ('report patient wise for: ',th,' with minroiarea:',minRoif)
            print ('0: no pneumonia , 1: pneumonia')
            fscore, acc, recall,pres, cmat=evaluate(y_true,y_pred,2)
            print ('classification_report:')
            print (classification_report(y_true, y_pred))
            print (cmat)
            print('Val F-score: '+str(fscore)+' Val acc: '+str(acc)+' Val recall: '+str(recall),' Val precision: '+str(pres))         
            tabz[iminroi][ith]=fscore

            if fscore>fsGood:
                fsGood=fscore
                roiGoodfs=minRoif
                thGoodfs=th
            
            if acc>accGood:
                accGood=acc
                roiGoodacc=minRoif
                thGoodacc=th
                
            if recall>recallGood:
                recallGood=recall
                roiGoodrecall=minRoif
                thGoodrecall=th

                
            print ( '---------------')
            filew=open(todaytrshFile,'a')
            
            filew.write('report patient wise for th: '+str(th)+' with minroiarea:'+str(minRoif)+'\n')    
            filew.write( '0: no pneumonia , 1: pneumonia\n')          
            filew.write('f-score is : '+ str(fscore)+'\n')
            filew.write('accuray is : '+ str(acc)+'\n')
            filew.write('recall is : '+ str(recall)+'\n')
            filew.write('precision is : '+ str(pres)+'\n')
            filew.write('confusion matrix\n')
            n= cmat.shape[0]
            for cmi in range (0,n):     
                            filew.write('  ')
                            for j in range (0,n):
                                filew.write(str(cmat[cmi][j])+' ')
                            filew.write('\n')
                        
            filew.write('------------------------------------------\n')
            filew.write('classification_report\n')
            filew.write(classification_report(y_true, y_pred)+'\n')
            filew.write('------------------------------------------\n')
            filew.close()
            if withScorePixel:
                print ('report pixel wise for threshold',th,'minroiarea:',minRoif)
               
                ysf = (predTot[:, :, :, 0] > th).flatten()
                yvf = (maskTot[:, :, :, 0] > 0.5).flatten()
                        
                fscore, acc, recall, pres, cmat = evaluate(yvf,ysf,2)
                iou=jaccard_similarity_score(yvf, ysf)
                print ('Val iou: ',iou)
                print ('classification_report:')
                print (classification_report(yvf, ysf))
                print (cmat)
                print('Val F-score: '+str(fscore)+' Val acc: '+str(acc)+' Val recall: '+str(recall),' Val precision: '+str(pres))         
                print ('---------------')
                
                filew=open(todaytrshFile,'a')
                
                filew.write('score for pixel for th: '+str(th)+'minroiarea:'+minRoif+'\n')          
                filew.write('f-score is : '+ str(fscore)+'\n')
                filew.write('accuray is : '+ str(acc)+'\n')
                filew.write('recall is : '+ str(recall)+'\n')
                filew.write('precision is : '+ str(pres)+'\n')
                filew.write('confusion matrix\n')
                n= cmat.shape[0]
                for cmi in range (0,n):     
                                filew.write('  ')
                                for j in range (0,n):
                                    filew.write(str(cmat[cmi][j])+' ')
                                filew.write('\n')
                            
                filew.write('------------------------------------------\n')
                filew.write('classification_report\n')
                filew.write(classification_report(yvf, ysf)+'\n')
                filew.write('------------------------------------------\n')
                filew.close()
    print ('summary:    ')
    print ('roi area for max acc',roiGoodacc)
    print ('roi area for max fs',roiGoodfs)
    print ('roi area for max recall',roiGoodrecall)
        
    print( 'th for max acc',thGoodacc)
    print( 'th for max fs',thGoodfs)
    print ('th for max recall',thGoodrecall)
        
    print ('max acc',accGood)
    print ('max fs',fsGood)
    print ('max recall'  ,recallGood )

    todaytrshFilePlot=os.path.join(reportDir,todayracine+'.png')  

    fig = plt.figure(figsize=(20,15))
    ax = fig.gca(projection='3d')
#    ax = plt.axes(projection='3d')
  
    
    surf = ax.plot_surface(tabx, taby, tabz, cmap=cm.coolwarm,
                           linewidth=1, antialiased=True,rstride=1,cstride=1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    ax.set_ylabel('th')
    ax.set_xlabel('roi area')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.legend(loc='lower right', shadow=True,fontsize=10)
    plt.savefig(todaytrshFilePlot)   
    plt.show()
    
    filew=open(todaytrshFile,'a')
    filew.write( 'summary: \n')    
        
    filew.write( 'max acc: '+str(accGood)+', th for max acc: '+str(thGoodacc)+', roi area for max acc: '+str(roiGoodacc)+'\n')
    filew.write( 'max fs: '+str(fsGood)+', th for max fs: '+str(thGoodfs)+', roi area for max fs: '+str(roiGoodfs)+'\n')
    filew.write( 'max recall: '  +str(recallGood) +', th for max recall: '+str(thGoodrecall)+', roi area for max recall: '+str(roiGoodrecall)+'\n')
    filew.write('----------------------\n')
    filew.close()
    
def test_images_pred(test_fns,th):

    pred_rle = []
    ids = []
    for f in tqdm_notebook(test_fns):
        img = pydicom.read_file(f).pixel_array
        img = resize(img,(img_rows,img_cols))
        img = model.predict(img.reshape(1,img_rows,img_cols,1))
        print(img.min(),img.max())
        
        img = img.reshape(img_rows,img_cols)
        ids.append('.'.join(f.split('/')[-1].split('.')[:-1]))
        #img = PIL.Image.fromarray(((img.T*255).astype(np.uint8)).resize(1024,1024))
        img = PIL.Image.fromarray((img.T*255).astype(np.uint8)).resize((1024,1024))
        img = np.asarray(img)
        #print(img)
        if img.max()>th:
            pred_rle.append(mask2rle(img,1024,1024))
        else:
             pred_rle.append('-1')
            
    return pred_rle,ids
        
def loadModelGlobal(kf):

    if kfold==0:
        kf=''
    f=open(todaytrshFile,'a')
    f.write('load model name: ' +modelName+'\n')
    print('load model name: ' +modelName)
    print ('----')
    if mergeDir:
        print  ('load',modelName)
        f.write('load model name: ' +modelName+'\n')
        listmodel=[name for name in os.listdir(os.path.join(cwd,repDir[modelName])) if name.find('.hdf5')>0] 
        if len(listmodel)>0:
            namelastc=load_weights_set(os.path.join(cwd,repDir[modelName]))  
            f.write('load weight: ' +namelastc+'\n')
        model=get_model(modelName,num_class,num_bit,img_rows,img_cols,True,weights,withWeight,namelastc,learning_rate,True,d0Dict[modelName])
        print ('----')
        
        print ( 'load',modelName2)
        f.write('load model name: ' +modelName2+'\n')
        listmodel=[name for name in os.listdir(os.path.join(cwd,repDir[modelName2])) if name.find('.hdf5')>0] 
        if len(listmodel)>0:
            namelastc=load_weights_set(os.path.join(cwd,repDir[modelName2]))  
            f.write('load weight: ' +namelastc+'\n')
        model2=get_model(modelName2,num_class,num_bit,img_rows,img_cols,True,weights,withWeight,namelastc,learning_rate,True,d0Dict[modelName2])
        print ( '----')
        
        print  ('load',modelName3)
        f.write('load model name: ' +modelName3+'\n')
        listmodel=[name for name in os.listdir(os.path.join(cwd,repDir[modelName3])) if name.find('.hdf5')>0] 
        if len(listmodel)>0:
            namelastc=load_weights_set(os.path.join(cwd,repDir[modelName3]))  
            f.write('load weight: ' +namelastc+'\n') 
        model3=get_model(modelName3,num_class,num_bit,img_rows,img_rows,True,weights,withWeight,namelastc,learning_rate,True,d0Dict[modelName3])
        print ('----')
        f.close()
        return model,model2,model3
    else:
        print ('load',modelName)
        namelastc='NAN'
        if usePreviousWeight:
            print('use previous weight')
            if os.path.exists(os.path.join(reportDir,dirToConsider,str(kf))):
                pathToWeight=os.path.join(reportDir,dirToConsider),str(kf)            
            
            elif os.path.exists(os.path.join(reportDir,str(kf))):
                    pathToWeight=os.path.join(reportDir,str(kf))
            else:
                pathToWeight=os.path.join(reportDir)

            listmodel=[name for name in os.listdir(pathToWeight) if name.find('.hdf5')>0] 
            if len(listmodel)>0:
                namelastc=load_weights_set(pathToWeight)  
                f.write('load weight: ' +namelastc+'\n')
                print('load weight: ' +namelastc)
            else:
                    print ('no weight at ',pathToWeight)
                    f.write('no weight at: '+pathToWeight+'\n')
        else:
                    print('do not use previous weight')
    
        model=get_model(modelName,num_class,num_bit,img_rows,img_cols,True,weights,withWeight,namelastc,learning_rate,True,d0)
        if writeModel:
            json_string = model.to_json()
            pickle.dump(json_string,open( modelName+'_'+str(img_rows)+'x'+str(img_cols)+'_CNN.h5', "wb"),protocol=-1)
        print ('d0: ',d0)
        f.close()
        return model

def withScoref0():
    th=0.51
    test_fns=sorted(glob(TEST_DCM_DIR+'*/*/*/*.dcm'))

    
    preds,ids = test_images_pred(test_fns[0:10],th)
    

    #print(preds[0])
    plt.imshow(rle2mask(preds[0],1024,1024))
    plt.show()

    
    #print(preds[10])
    print(len(preds),len(ids))
    

    
    submission = pd.DataFrame({'ImageId':ids,'EncodedPixels':preds})
    

    
    print(submission.head())

    
    submission.to_csv('submission.csv',index = False)
    
def onePredf():
    random.seed(42)
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
        
#    train_filenames,valid_filenames=getfilenames()

    vf=['1.2.276.0.7230010.3.1.4.8323329.5577.1517875188.867087.dcm','1.2.276.0.7230010.3.1.4.8323329.10012.1517875220.965942.dcm']
#    vf=['1.2.276.0.7230010.3.1.4.8323329.10012.1517875220.965942.dcm']

    for filename in vf:

        print (filename)
        imageDest=os.path.join('../input/refDicom',filename)
    
        imgt,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(imageDest,img_rows,img_cols,dirWriteBmp='',
                                                writeBmp=False,resize=True,histAdapt=histAdapt)
        print(MIN_BOUND,MAX_BOUND)
        print('imgt min max',imgt.shape,imgt.min(),imgt.max())

        img=normAdapt(imgt,MIN_BOUND,MAX_BOUND,zerocenter)

        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        print(img.shape,img.min(),img.max())

        pred = model.predict(img,verbose=1)
        pred=pred[0].reshape(img_rows,img_cols)

#        pred=np.squeeze(pred[0])

#        pred= cv2.resize(pred,(img_rows,img_cols),interpolation=cv2.INTER_NEAREST)  
        print('pred min max',pred.shape,pred.min(),pred.max())
#        msk=maskfromrle(df_full,imageDest,img_rows,img_cols,rle2mask)
        if type(df_full.loc[filename[:-4],' EncodedPixels']) == str:
                msk = rle2mask(df_full.loc[filename[:-4],' EncodedPixels'], 1024, 1024)
                    
        else:
                msk= np.zeros((1024, 1024),np.uint8)
                for x in df_full.loc[filename[:-4],' EncodedPixels']:                      
                    msk =  np.clip(msk + rle2mask(x, 1024, 1024),0,255)
        msk=cv2.resize(msk,(img_rows,img_cols),interpolation=cv2.INTER_NEAREST)  
        msk=msk.T
#        plt.imshow(normi(imgt),cmap = 'gray')
#        plt.show()
#        plt.imshow(normi(msk))
#        plt.show()
#        imgt=cv2.resize(img,(img_rows,img_cols))
        mkadd=cv2.addWeighted(normi(imgt),0.5,normi(msk),0.5,0)
        plt.imshow(normi(mkadd))
        plt.show()
        for thresh in prob_thresholds:
            print(thresh)
            pred_to_aff = pred.copy()
            np.putmask(pred_to_aff,pred <= thresh,0)
            np.putmask(pred_to_aff,pred > thresh,1)
#            print(pred_to_aff.min(),pred_to_aff.max())
    

            plt.imshow(pred_to_aff)
            plt.show()
#            plt.hist(pred_to_aff, bins=100)
##        axs[1].hist(pred, bins=n_bins)
#            plt.show


#        plt.hist(pred, bins=n_bins)
##        axs[1].hist(pred, bins=n_bins)
#        plt.show
#        np.putmask(pred,pred > th,255)
#        np.putmask(pred,pred <= th,0)
#        pred = cv2.dilate(pred, kernel,iterations=3) 
#        pred = cv2.erode(pred, kernel,iterations=3)
#        pred = cv2.dilate(pred, kernel,iterations=3) 
#        pred = cv2.erode(pred, kernel,iterations=3) 
    #    comp = pred[0] > 0.5
                # apply connected components
        
            # get filename without extension
#        filename = filename.split('.')[0]


#        cv2.imwrite(filename+str(numb)+'mask.jpg',mask)
#        cv2.imwrite(filename+str(numb)+'pred.jpg',pred)
            
            
            
def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.
        
    Returns:
        dict: contains metadata of relevant fields.
    """   
            
    data = {}
    
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data
def get_test_tensor(file_path, batch_size, img_size, channels):
        imgt,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(file_path,img_size,img_size,dirWriteBmp='',
                                                writeBmp=False,resize=True,histAdapt=histAdapt)
        img=normAdapt(imgt,MIN_BOUND,MAX_BOUND,zerocenter)

        img = np.expand_dims(img, -1)
        X = np.expand_dims(img, 0)
    
#        X = np.empty((batch_size, img_size, img_size, channels))
#
#        # Store sample
#        pixel_array = pydicom.read_file(file_path).pixel_array
#        image_resized = cv2.resize(pixel_array, (img_size, img_size))
#        image_resized = np.array(image_resized, dtype=np.float64)
#
#        image_resized -= image_resized.mean()
#        image_resized /= image_resized.std()
#        X[0,] = np.expand_dims(image_resized, axis=2)

        return X
def withScoref0():
    test_fns=sorted(glob(TEST_DCM_DIR+'*/*/*/*.dcm'))
    rles_df = pd.read_csv(RAW_TRAIN_LABELS)
# parse test DICOM dataset
    test_metadata_df = pd.DataFrame()
    test_metadata_list = []
    for file_path in tqdm(test_fns):
        dicom_data = pydicom.dcmread(file_path)
        test_metadata = dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=False)
        test_metadata_list.append(test_metadata)
    test_metadata_df = pd.DataFrame(test_metadata_list)
    submission = []
    for i, row in test_metadata_df.iterrows():

        test_img = get_test_tensor(test_metadata_df['file_path'][i],1,img_rows,1)
        
        pred_mask = model.predict(test_img,verbose=2).reshape((img_rows,img_cols))
#        print(pred_mask.shape,pred_mask.min(),pred_mask.max())
        prediction = {}
        prediction['ImageId'] = str(test_metadata_df['id'][i])
        pred_mask = (pred_mask > .5).astype(int)
#        print(str(test_metadata_df['id'][i]))
        
        if pred_mask.sum() < 1:
            prediction['EncodedPixels'] =  -1
        else:
            prediction['EncodedPixels'] = mask2rle(pred_mask * 255, img_rows, img_cols)
        submission.append(prediction)
    submission_df = pd.DataFrame(submission)
    submission_df = submission_df[['ImageId','EncodedPixels']]
    print(submission_df.head())
    fileResult=os.path.join(reportDir,str(today)+'_submission.csv')

    submission_df.to_csv(fileResult, index=False)
    
    
def withScoref():
    
    test_fns=sorted(glob(TEST_DCM_DIR+'*/*/*/*.dcm'))
    lvfn=len(test_fns)
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
    batch_size=32
    valid_gen = generator(test_fns,df_full, pneumothorax, 
                              batch_size=batch_size, image_size=(img_rows,img_cols), 
                              shuffle=False, augment=False, predict=True)
    indb=-1
    for imgs, fnames in valid_gen:
        indb+=1
        preds = model.predict(imgs,batch_size=batch_size,verbose=1)
        if indb==0:
            predTot=preds.copy()
            maskTot=fnames.copy()
        else:
            predTot=np.concatenate((predTot,preds),axis=0)
            maskTot=maskTot+fnames
        if predTot.shape[0]>=lvfn:
            predTot=predTot[:lvfn]
            maskTot=maskTot[:lvfn]
            break

    max_images = 64
    grid_width = 16
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    # for i, idx in enumerate(index_val[:max_images]):
    for i, idx in enumerate(test_fns[:max_images]):
        img ,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(idx,img_rows,img_cols,dirWriteBmp='',
                                                writeBmp=False,resize=True,histAdapt=histAdapt)
        pred = predTot[i].squeeze()
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(np.array(np.round(pred > thForScore), dtype=np.float32), alpha=0.5, cmap="Reds")
        ax.axis('off')
    
    
    
    rles = []
    i,max_img = 1,10
    plt.figure(figsize=(16,4))
    for p in tqdm_notebook(predTot):
        p = p.squeeze()
        im = cv2.resize(p,(1024,1024))
        im = im > thForScore
#         zero out the smaller regions.
        if im.sum()<1024*2:
            im[:] = 0
        im = (im.T*255).astype(np.uint8)  
        rles.append(mask2rle(im, 1024, 1024))
        i += 1
        if i<max_img:
            plt.subplot(1,max_img,i)
            plt.imshow(im)
            plt.axis('off')
    ids = [o.split('/')[-1][:-4] for o in maskTot]
    sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    print(sub_df.head())
    fileResult=os.path.join(reportDir,str(today)+'_submission.csv')
    sub_df.to_csv(fileResult, index=False)
        
    
def trialaugm():
        train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()
        for filename in train_filenames[0:5]:

#            print (filename)        
            img,correct,MIN_BOUND,MAX_BOUND,imgShape=geneTable(filename,img_rows,img_cols,dirWriteBmp='',
                                                    writeBmp=False,resize=True,histAdapt=histAdapt)
           
            print('1',img.shape,img.min(),img.max())

            filename = filename.split('/')[-1]

    
            if type(df_full.loc[filename[:-4],' EncodedPixels']) == str:
                    msk = rle2mask(df_full.loc[filename[:-4],' EncodedPixels'], 1024, 1024)
                        
            else:
                    msk= np.zeros((1024, 1024),np.uint8)
                    for x in df_full.loc[filename[:-4],' EncodedPixels']:                      
                        msk =  np.clip(msk + rle2mask(x, 1024, 1024),0,255)
            msk=cv2.resize(msk,(img_rows,img_cols),interpolation=cv2.INTER_NEAREST)  
            msk=msk.T
            print('1',msk.shape,msk.min(),msk.max())
            mkadd=cv2.addWeighted(normi(img),0.5,normi(msk),0.5,0)
            plt.imshow(normi(mkadd))
            plt.show()

            addint,multint,rotimg,resiz,shiftv,shifth,rotateint=generandom(maxadd,
                                    maxmult,maxrot,maxresize,maxshiftv,maxshifth,maxrotate,keepaenh)
            img=geneaug(img,addint,multint,rotimg,resiz,shiftv,shifth,rotateint,False,img.min(),img.max())
            msk=geneaug(msk,  0,    0,     rotimg,resiz,shiftv,shifth,rotateint,True,msk.min(),msk.max())
            print('2',img.shape,img.min(),img.max())
            print('2',msk.shape,msk.min(),msk.max())

            mkadd=cv2.addWeighted(normi(img),0.5,normi(msk),0.5,0)
            plt.imshow(normi(mkadd))
            plt.show()

def withPlotf():
    print('with plot 32')
    random.seed(42)
#    train_filenames,valid_filenames,maskRoi=getfilenames()
    train_filenames,valid_filenames,pneumothorax,df_full=getfilenames()

    print()
    areamask=[]
    areapred=[]
    print ('blue: true  red: predict on 32 images')
    valid_gen = generator(train_filenames,df_full, pneumothorax, 
                              batch_size=32, image_size=(img_rows,img_cols), 
                              shuffle=False, augment=False, predict=False)
#    valid_gen = generator(folderTrain, valid_filenames, pneumothorax, 
#                          batch_size=32, image_size=img_rows, shuffle=False, augment=False, predict=False)
    ind=0
    for imgs, msks in valid_gen:
#        print ind
        if ind>=31:
                break
        # predict batch of images
        preds = model.predict(imgs,verbose=1, batch_size=batchSize[modelName])
        if mergeDir:          
            preds2 = model2.predict(imgs,batch_size=batchSize[modelName2],verbose=1)     
            preds3 = model3.predict(imgs,batch_size=batchSize[modelName3],verbose=1)    
            preds=(preds+preds2+preds3)/3
            del preds2,  preds3

    
        # create figure
        f, axarr = plt.subplots(4, 8 ,figsize=(20,15))
        axarr = axarr.ravel()
        axidx = 0
        # loop through batch
        for img, msk, pred in zip(imgs, msks, preds):
            # plot image
            axarr[axidx].imshow(img[:, :, 0])
            # threshold true mask
            comp = msk[:, :, 0] > 0.5

            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                areamask.append(width*height)

                axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='b',facecolor='none'))
            # threshold predicted mask
            comp = pred[:, :, 0] > 0.5

            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                areapred.append(width*height)
                axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',facecolor='none'))
            axidx += 1
            ind+=1
        plt.show()
        
        if ind>=31:
            break
#
def calparamf():
    
    print('calculate param')
    file_train=sorted(glob(TRAIN_DCM_DIR+'*/*/*/*.dcm'))
    
    #train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
    #test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))
    
    print(len(file_train))
    pneumothorax=[]
    areamask=[]

#df = pd.read_csv(RAW_TRAIN_LABELS, header=None, index_col=0)
    df_full = pd.read_csv(RAW_TRAIN_LABELS, index_col='ImageId')
    
    for n, _id in tqdm_notebook(enumerate(file_train), total=len(file_train)):
        try:
                if not '-1'  in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
        #            Y_train[n] = np.zeros((1024, 1024, 1))
                    pneumothorax.append(_id.split('/')[-1])
        except KeyError:
            pass

    print  ( 'total number of images:',len(file_train))
    for filename in  file_train:
        filename = filename.split('/')[-1]
            # if image contains pneumonia
        if filename in pneumothorax:
                msk=maskfromrle(df_full,filename,img_rows,img_cols,rle2mask)
                mskm=msk>128
                comp = measure.label(mskm)
            # apply bounding boxes
                for region in measure.regionprops(comp):
                    # retrieve x, y, height and width
                    y, x, y2, x2 = region.bbox
                    height = y2 - y
                    width = x2 - x
                    areamask.append(width*height)

    ordlistft=sorted(areamask,reverse=False)
    for i in range(5):
        print ('minimum areaa',i,ordlistft[i])
    for i in range(5):
        print ('maximum area',i,ordlistft[-(1+i)])
        
if __name__ == '__main__':
#    trialaugm()
#    ooo
    if not withTrain and not calparam:
        if mergeDir:
                model,model2,model3=loadModelGlobal(kfold)
        else:
                model=loadModelGlobal(kfold)
   
    if lookForLearningRate:
        lookForLearningRatef()
        
    if withScore:
        withScoref()
    if withTrain:
        withTrainf()

    if withPlot32:
        withPlotf()
        
    if calparam:
        calparamf()
        
    if withReport:
        withReportf()
    
    if withEval:
        withEvalf()
    if onePred:
        onePredf()
           
        
    f=open(todaytrshFile,'a')
    spenttime=spentTimeFunc(time.time() - start)
    print ('total time spent :' ,spenttime)  
    f.write('total time spent: '+str(spenttime)+'\n')
        
    f.close()


#
##preds_t = (preds_test>best_thr).long().numpy()
#rles = []
#for p in preds:
#    
#    print(p.find('-1'))
#
##    if(p.find('-1') !=-1):
##        im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))
##        im = np.asarray(im)
##        rles.append(mask2rle(im, 1024, 1024))
##    else: rles.append('-1')
#    
#sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
#sub_df.to_csv('submission1.csv', index=False)
#

#
#from IPython.display import HTML
#html = "<a href = submission.csv>d</a>"
#HTML(html)


# <h4>Work in progress. I will update it.</h4>
# <h3>Any suggestion. Let me know in comments.</h3>
# 
