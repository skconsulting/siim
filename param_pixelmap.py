# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 14:46:39 2018
set of utilities for all programs
when run in main, generates directories from classes

@author: sylvain Kritter
"""
import pickle
import numpy as np
import os
import random
import pydicom
import cv2
import shutil
import time
import sys
#import math
import collections



#from tensorflow import keras
from keras.utils import np_utils
import sklearn.metrics as metrics
#import skimage
#from skimage.filters import *
import skimage.exposure as exposure
#modelName='unet'



keepaenh=1#0 no augmentatiuon, 1 augmentation
#max limit for random value for enhancement 
maxmult=0 #multiply pixel value in % => 10 = 1.1
maxadd=0 #add data to pixel value in % of max range
maxrot=1 #rot + flip max 7 1 for only lrflip
maxresize=5 #resize factor in %  ==> 20
maxshiftv=2 #shift vertical in percentage of height ==>5 
maxshifth=2 #shift horizontal in percentage of width   ==>5
maxrotate=5 #degre rotation of image ==>5


#PIXEL_MEAN=0.175
reservedWord=[]
black=(0,0,0)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255,255,0)
cyan=(0,255,255)
purple=(255,0,255)
white=(255,255,255)
darkgreen=(11,123,96)
pink =(255,128,255)
lightgreen=(125,237,125)
orange=(255,153,102)
lowgreen=(0,51,51)
parme=(234,136,222)
chatain=(139,108,66)


classNameRefIncidences='classIncidencesTestPoignet.pkl'
classNameRefPatho='classPathoTestPoignet.pkl'
classifnotvisu=[]
classPathoF ={
        'healthy':0,
        'path1':1, #rectangle
        'path2':2, #cercle
        'path3':3 #ellipse
        }
classPathoS ={
        'healthy':0,
        'path1':1
#        , #rectangle
#        'path2':2, #cercle
#        'path3':3 #ellipse
        }
classPatho=classPathoS
classifc ={
    'healthy':black,
    'path1':blue,
    'path2':red,
    'path3':yellow
}

#functions
def get_class_weights(y):
    counter = collections.Counter(y)
    majority = 1.0*max(counter.values())
    weights=np.zeros((len(counter)),np.float32)
    for i in range(len(counter)):
        weights[i]=majority/counter[i]
    return  weights

def numbclasses(y):
    y_train=np.array(y)
    uniquelbls = np.unique(y_train)
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes in this data:', int(nb_classes)) 
    nb_classes = len( classPatho)
    print ('number of classes in this set:', int(nb_classes)) 
    y_flatten=y_train.flatten()
    class_weights= 1.0*get_class_weights(y_flatten)    
    return class_weights

def formatData(X,y,numclass,num_bit_):
    """format list data (X) and tag(y) for cnn"""
    y_tab= np.asarray(y)

    y_cat=np.zeros((y_tab.shape[0],y_tab.shape[1],y_tab.shape[2],numclass),np.uint8)

    x_tab= np.asarray(X)
    if num_bit_==3:
        x_tab=np.repeat(x_tab[:,:,:,np.newaxis],3,axis=-1)
    else:
        x_tab = np.expand_dims(x_tab,-1)  

    for i in range (y_tab.shape[0]):
        for j in range (0,y_tab.shape[1]):
#            print i,j,y_test[i][j]
            y_cat[i][j] = np_utils.to_categorical(y_tab[i][j], numclass)

    return  x_tab, y_cat


def load_weights_set(pickle_dir_train,str2ch):
    """ load last weights"""
    listmodel=[name for name in os.listdir(pickle_dir_train) if name.find('.hdf5')>0 and name.find(str2ch)>0 ]
#    print 'load_model',pickle_dir_train
    ordlist=[]
    for name in listmodel:
        nfc=os.path.join(pickle_dir_train,name)
        nbs = os.path.getmtime(nfc)
        tt=(name,nbs)
        ordlist.append(tt)
    ordlistc=sorted(ordlist,key=lambda col:col[1],reverse=True)
    namelast=ordlistc[0][0]
    namelastc=os.path.join(pickle_dir_train,namelast)
#    print 'last weights :',namelast   
    return namelastc


def affiche(nom,image):
    cv2.imshow(nom,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def spentTimeFunc(tt):
  days=int(tt/(3600*24))
  hours=int((tt-(days*3600*24))/3600)
  minutes=int((tt-(days*3600*24)-(hours*3600))/60)
  seconds=int(tt%60)
  return str(days)+'days '+str(hours)+'hours '+str(minutes)+'minutes '+str(seconds)+'seconds'
    
def geneTable(filePath,img_rows_,img_cols_,dirWriteBmp='',writeBmp=False,resize=True,histAdapt='NAN'):
    """generates table from dicom, resize to cols and rows dimensions, 
    applies normalisation"""

    (top,tail)=os.path.split(filePath)
    nameBmp=tail[0:tail.lower().find('.dcm')]+'.png'
    (top1,tail1)=os.path.split(top)
    if writeBmp and not os.path.exists(dirWriteBmp):
        os.mkdir(dirWriteBmp)

    correct=True
    tabPixelArray=[]
    
    RefDs = pydicom.dcmread(filePath,force=True)
    

    rows=RefDs.Rows
    columns=RefDs.Columns
#            print 'rows',rows,'columns',columns
    BS= RefDs.BitsStored
    BA= RefDs.BitsAllocated
    HB= RefDs.HighBit
    SPP= RefDs.SamplesPerPixel
#    print BS,BA,HB,SPP
#    dsr= RefDs.pixel_array
     
    try:
        PR= RefDs.PixelRepresentation
#        print RefDs.PixelRepresentation
    except:
        PR=0
    try:       
        dsr= RefDs.pixel_array
#        print 'pixel_array'
    except:
        dsr= RefDs.PixelData
        
        if BA==8:
                dsr=np.frombuffer(dsr,dtype=np.uint8)
        else:
                dsr=np.frombuffer(dsr,dtype=np.uint16)

        dsr=dsr[1:]

        if SPP==3:
            dsr=np.reshape(dsr,(rows,columns,3))
        else:
            dsr=np.reshape(dsr,(rows,columns))
    if SPP==3:
            dsr=dsr[:,:,0]
#    print 'before',dsr.shape,dsr.min(),dsr.max()
    dsr=dsr.astype(np.uint16) 
    dsrshape=dsr.shape
    if 2**BS<dsr.max():
#shift left to eliminate head bits
        dsr=np.left_shift(dsr,BA-HB-2)       
    #shift right to eliminate trailing bits           
        dsr=np.right_shift(dsr,BA-BS-1)
    else:
        dsr=np.left_shift(dsr,BA-HB-1)       
    #shift right to eliminate trailing bits           
        dsr=np.right_shift(dsr,BA-BS)
    
    if PR==1:
            dsr=dsr.astype(np.int16)
    else:
            dsr=dsr.astype(np.uint16)    
#    print 'after',dsr.shape,dsr.min(),dsr.max()
    
    try:
            PixelPaddingValue= RefDs.PixelPaddingValue
            dsr[dsr == PixelPaddingValue] = dsr.min()
    except:
                pass
#    print ('after',dsr.shape,dsr.min(),dsr.max())
    npmax= np.count_nonzero(dsr== dsr.max())
    npmax1= np.count_nonzero(dsr== dsr.max()-1)

    npmin= np.count_nonzero(dsr== dsr.min())
    npmin1= np.count_nonzero(dsr== dsr.min()-1)

    if npmax1==0 and npmax>0:  
        maxVal= dsr.max()
        dsrm=dsr.mean()
        np.putmask(dsr,dsr==maxVal,dsrm)
        
    if npmin1==0 and npmin>0:
        minval= dsr.min()
        dsrm=dsr.mean()
        np.putmask(dsr,dsr==minval,dsrm)


    if RefDs.PhotometricInterpretation=='MONOCHROME1':
#        print 'MONOCHROME1'
        dsrMax=dsr.max()
        dsr=dsrMax-dsr
#    print ('after',dsr.shape,dsr.min(),dsr.max())
    #rescale, intercept
    try:
        intercept = RefDs.RescaleIntercept
#        print ('intercept',intercept)
    except:
        intercept=0   
    try:
        slope = RefDs.RescaleSlope
#        print ( 'slope',slope)
    except:
            slope=1
    typedsr=dsr.dtype
    if slope != 1:
        dsr = slope * dsr.astype(np.float32)
        dsr = dsr.astype(np.int16)
    dsr += np.int16(intercept)
    dsr=dsr.astype(typedsr)
#    print ('after slope',dsr.shape,dsr.min(),dsr.max())
#    MAX_BOUND=2**BS

    if resize and (img_rows_,img_cols_)!=dsrshape:
        dsr = dsr.astype('float32')
        #resize image to the maximum, keeping aspect ration
        fx=img_rows_/rows
        fy=img_cols_/columns
        rf=min(fx,fy)
        xmin=dsr.min()
        xmax=dsr.max()
        dsr=cv2.resize(dsr,None,fx=rf,fy=rf,interpolation=cv2.INTER_AREA)
        dsr=np.clip(dsr,xmin,xmax)
        dsr=dsr.astype(typedsr)
        #Center images, define a black image of total size
        dsrColumn=dsr.shape[1]
        dsrRows=dsr.shape[0]
        tabPixelArray=np.zeros((img_rows_,img_cols_),typedsr)
        tabPixelArray[:,:]=dsr.min()
        #put in the center the image
        dsrColC=int((img_cols_-dsrColumn)/2.0)
        dsrRowC=int((img_rows_-dsrRows)/2.0)
        tabPixelArray[dsrRowC:dsrRows+dsrRowC,dsrColC:dsrColumn+dsrColC]=dsr
  
    else:
        tabPixelArray=dsr.copy()
#    print ('after resize',tabPixelArray.shape,tabPixelArray.min(),tabPixelArray.max())

#    tabPixelArray=dsr.astype(np.int16)
    if writeBmp:
            writedir=os.path.join(dirWriteBmp,nameBmp)

            bmpFile=normi(tabPixelArray)
            cv2.imwrite(writedir,bmpFile)    
#     # Contrast stretching
#    p2, p98 = np.percentile(tabPixelArray, (2, 98))
#    tabPixelArray = exposure.rescale_intensity(tabPixelArray, in_range=(p2, p98))    
# Equalization
#    tabPixelArray = exposure.equalize_hist(tabPixelArray)  
    if histAdapt=='histAdapt':
        tabPixelArray=exposure.equalize_adapthist(tabPixelArray,kernel_size=128,clip_limit=0.03) #defalut 128
    elif histAdapt=='clahe':
        clahe=cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        tabPixelArray=clahe.apply(tabPixelArray)
        
    elif histAdapt != 'NAN':
        print('error histadapt')
        sys.exit()
        

#    print ('after histAdapt',tabPixelArray.shape,tabPixelArray.min(),tabPixelArray.max())

# Adaptive Equalization
#    tabPixelArray = exposure.equalize_adapthist(tabPixelArray, clip_limit=0.03)   
    minb =tabPixelArray.min()
    maxb =tabPixelArray.max()
#    if tabPixelArray.max()==0:
#            maxb=0
#    elif tabPixelArray.max()>0:
#        loga=math.log(tabPixelArray.max(),2)
#        ceilloga=math.ceil(loga)
#        maxb=int(2**ceilloga)-1
#        if maxb==tabPixelArray.max()-1:
#            maxb=int(2**(ceilloga+1))-1
#    else:
#        loga=math.log(-tabPixelArray.max(),2)
#        intloga=int(loga)
#        maxb=-int(2**(intloga))+1
#        if maxb==(tabPixelArray.max()/2)+1:
#            maxb=-int(2**(intloga+1))+1
# 
#       
#    if tabPixelArray.min()==0:
#        minb=0
#
#    elif mint>0:  
#        loga=math.log(mint,2)
#        ceilloga=math.ceil(loga)
#        print 'ceilloga',ceilloga
#        print '2pm1',2**(-1)
#        minb=int(2**(ceilloga-1)-1)
#        
#        if minb==(mint-1)/2:
#            minb=int(2**(ceilloga))-1  
#        print 'minb',minb
#                 
#    else:
#        loga=math.log(-minb,2)
#        intloga=int(loga)
#        minb=-int(2**(intloga+1))+1
#        if minb==(minb/2)+1:
#            minb=-int(2**(intloga+1))+1    
#      
             
    return tabPixelArray,correct,minb,maxb,dsrshape




def geneTable0(filePath,img_rows_,img_cols_,dirWriteBmp='',writeBmp=False,resize=True,histAdapt='NAN'):
    """generates table from dicom, resize to cols and rows dimensions, 
    applies normalisation"""

    (top,tail)=os.path.split(filePath)
    nameBmp=tail[0:tail.lower().find('.dcm')]+'.png'
    (top1,tail1)=os.path.split(top)
    if writeBmp and not os.path.exists(dirWriteBmp):
        os.mkdir(dirWriteBmp)

    correct=True

    image = pydicom.read_file(filePath).pixel_array
    dsrshape=image.shape
    if resize: 
        image = cv2.resize(image, (img_rows_, img_cols_))
    
# Equalization
#    tabPixelArray = exposure.equalize_hist(tabPixelArray)  
    if histAdapt=='histAdapt':
        image=exposure.equalize_adapthist(image,kernel_size=128,clip_limit=0.03) #defalut 128
    elif histAdapt=='clahe':
        clahe=cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        image=clahe.apply(image)
        
    elif histAdapt != 'NAN':
        print('error histadapt')
        sys.exit()
        

#    print ('after histAdapt',tabPixelArray.shape,tabPixelArray.min(),tabPixelArray.max())

# Adaptive Equalization
#    tabPixelArray = exposure.equalize_adapthist(tabPixelArray, clip_limit=0.03)   
    minb =image.min()
    maxb =image.max()
    if writeBmp:
            writedir=os.path.join(dirWriteBmp,nameBmp)

            bmpFile=normi(image)
            cv2.imwrite(writedir,bmpFile)  

    return image,correct,minb,maxb,dsrshape

def normalizeCoef(image,minC,maxC):
    image1= (image - minC) / (maxC - minC)
    return image1

def zero_center(image):
#    image1 = image - PIXEL_MEAN #01
    image1 = image - image.mean() #02
#    image1 = (image - image.mean())/image.std() #03
#    image1 = (image - image.mean()) #04
#    image1 = image1/image1.std() #04
    return image1

def normAdapt(image,minC,maxC,z):
    imagef=image.astype('float32')
    image1=normalizeCoef(imagef,minC,maxC)
    if z: image1=zero_center(image1)
    return image1
def colorimage(image,color):
#    im=image.copy()
    im = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    np.putmask(im,im>0,color)
    return im
    
def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
#         print 'path exist'
         # remove if exists
         shutil.rmtree(path)
         time.sleep(1)
 
def fidclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():

        if valeur == numero:
            found=True
            return cle
    if not found:
        return 'unknown'
def evaluate(actual,pred,num_class):
    print ('fscore')
    fscore = metrics.f1_score(actual, pred, average='weighted')
    print ('accuracy')
    acc = metrics.accuracy_score(actual, pred)
    print ('precision')
    pres = metrics.precision_score(actual, pred,average='weighted')
    print ('recall')
    recall = metrics.recall_score(actual, pred, average='weighted')
    labl=[]
    for i in range(num_class):
        labl.append(i)
    print ('cm')
    cm = metrics.confusion_matrix(actual,pred,labels=labl)
    fscoref=round(fscore,3)
    accf=round(acc,3)
    recallf=round(recall,3)
    presf=round(pres,3)
    return fscoref, accf, recallf,presf, cm

def findclass(numero,classn):
    """return class from number"""
    found=False
#    print numero
    for cle, valeur in classn.items():
        if valeur == numero:
            found=True
            return cle
    if not found:
        return 'unknown'


def normi(tabi):
     """ normalise patches"""
#     tabi2=bytescale(tabi, low=0, high=255)
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
     mm=max_val-min_val
     if mm ==0:
         mm=1
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(255/mm)
     tabi2=tabi2.astype('uint8')
     return tabi2


def geneshiftv(img,s,MIN_BOUND):
    """vertical shift"""
    shap0=img.shape[0]
#    print MIN_BOUND
    shi=int(shap0*s/100.)
    if shi!=0:
        tr=np.empty_like(img)
        tr[:,:]=MIN_BOUND
        if shi>0:
            tr[shi:]=img[:-shi]           
        else:
            tr[:shi]=img[-shi:]

    else:
        tr=img
#    print 'inside',type(tr[0][0]),tr.shape,tr.min(),tr.max()

    return tr

def geneshifth(img,s,MIN_BOUND):
    "horizontal shift"""
    shap1=img.shape[1]
    shi=int(shap1*s/100.)  
    if shi!=0:
        tr=np.empty_like(img)
        tr[:,:]=MIN_BOUND
        if shi>0:
             tr[:,shi:]=img[:,:-shi]            
        else:
            tr[:,:shi]=img[:,-shi:]          
    else:
        tr=img
    return tr

def generesize(img,r,nearest,MIN_BOUND,MAX_BOUND):
    """resize image"""
    if r !=0:
        shapx=img.shape[1]
        shapy=img.shape[0]  
        types=img.dtype
        imgc=img.astype('float32')
        if nearest:
            imgr=cv2.resize(imgc,None,fx=(100+r)/100.,fy=(100+r)/100.,interpolation=cv2.INTER_NEAREST)  
        else:
            imgr=cv2.resize(imgc,None,fx=(100+r)/100.,fy=(100+r)/100.,interpolation=cv2.INTER_AREA)  
        imgr=np.clip(imgr,MIN_BOUND,MAX_BOUND)
        imgr=imgr.astype(types)
        newshapex=imgr.shape[1]
        newshapey=imgr.shape[0]
    
        if newshapex>shapx:
            dx=int((newshapex-shapx)/2)
            dy=int((newshapey-shapy)/2)
            imgrf=imgr[dy:dy+shapy,dx:dx+shapx]
        else:
            dx=int((shapx-newshapex)/2)
            dy=int((shapy-newshapey)/2)
            imgrf=np.empty_like(img)
            imgrf[:,:]=MIN_BOUND
            imgrf[dy:dy+newshapey,dx:dx+newshapex]=imgr
    else:
            imgrf=img
    return imgrf

def generot(image,tt):
    """ rotate and flip images (mulitple of 90 deg)"""
    if tt==0:
        imout=image
    elif tt==1:
    #1 flip fimage left-right
        imout=np.fliplr(image)
    elif tt==2:
    #2 180 deg
        imout = np.rot90( image,2)
    elif tt==3:
    #3 270 deg
        imout = np.rot90(image,3)
    elif tt==4:
    # 4 90 deg
        imout = np.rot90(image,1)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.fliplr(image),2)
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.fliplr(image),3)
    return imout


def geneadd(img,s,MIN_BOUND,MAX_BOUND):
    """add or substract constant to image"""
    if s!=0:
        types=img.dtype
     
        acts=s*(MAX_BOUND-MIN_BOUND)/100.0  
        imgr=img+acts
        imgr=np.clip(imgr,MIN_BOUND,MAX_BOUND)
        imgr=imgr.astype(types)
    else:
        imgr=img
    return imgr


def genemult(img,s,MIN_BOUND,MAX_BOUND):
    """multiply images by fix value"""
    if s!=0:      
        types=img.dtype
        acts=(100+s)/100.0  
        imgr=img*acts
        imgr=np.clip(imgr,MIN_BOUND,MAX_BOUND)
        imgr=imgr.astype(types)
    else:
        imgr=img
    return imgr


def generotate(img,s,nearest,MIN_BOUND,MAX_BOUND):
    if s!=0:
        types=img.dtype
#        print(types)
#        imgc=img.astype('float64')
        imgc=img.copy()

#        print('d1',imgc.min(),imgc.max())
        rows,cols = img.shape[0],img.shape[1]
        imgr=np.empty_like(imgc)
        imgr[:,:]=1.0*MIN_BOUND
        M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)),s,1)       
        if nearest:
            imgc=cv2.warpAffine(imgc,M,(cols,rows),imgr,cv2.INTER_NEAREST,cv2.BORDER_TRANSPARENT)
        else:
            imgc=cv2.warpAffine(imgc,M,(cols,rows),imgr,cv2.INTER_AREA,cv2.BORDER_TRANSPARENT)      
#        print('d2',imgr.min(),imgr.max())

        imgr=np.clip(imgr,MIN_BOUND,MAX_BOUND)
#        print('d3',imgr.min(),imgr.max())

        imgr=imgr.astype(types)
#        print('d4',imgr.min(),imgr.max())

    else:
        imgr=img
    return imgr


def geneaug(img,addint,multint,rotimg,resiz,shiftv,shifth,rotateint,nearest,MIN_BOUND,MAX_BOUND):
    """augment images with all functions"""
    imgr=generotate(img,rotateint,nearest,MIN_BOUND,MAX_BOUND)
    imgr=geneshifth(imgr,shifth,MIN_BOUND)  
    imgr=geneshiftv(imgr,shiftv,MIN_BOUND)
    imgr=generot(imgr,rotimg)
    imgr=generesize(imgr,resiz,nearest,MIN_BOUND,MAX_BOUND)
    imgr=genemult(imgr,multint,MIN_BOUND,MAX_BOUND)
    imgr=geneadd(imgr,addint,MIN_BOUND,MAX_BOUND)
    
    return imgr
    

def generandom(_maxadd,_maxmult,_maxrot,_maxresize,_maxshiftv,_maxshifth,_maxrotate,_keepaenh):
    """generate random value for augmentation in fixed range
    """

    addint =_keepaenh*_maxadd*random.uniform(-1.0, 1.0)

    multint =_keepaenh*_maxmult*random.uniform(-1.0, 1.0)

    rotimg =_keepaenh*random.randint(0, _maxrot)
    
    resiz =_keepaenh*_maxresize*random.uniform(-1.0, 1.0)
    
    shiftv =_keepaenh*_maxshiftv*random.uniform(-1.0, 1.0)

    shifth =_keepaenh*_maxshifth*random.uniform(-1.0, 1.0)

    rotateint =_keepaenh*_maxrotate*random.uniform(-1.0, 1.0)

    return addint,multint,rotimg,resiz,shiftv,shifth,rotateint



if __name__ == "__main__":
    #create directories according to classes
#    cwd=os.getcwd()
#    (cwdtop,tail)=os.path.split(cwd)
#    (cwdtop,tail)=os.path.split(cwdtop)
#
#    dataPatDLocM='patientDirectory_Trial'
#    dataPatDLoc=os.path.join(cwdtop,dataPatDLocM)
#    print (dataPatDLoc)
#    
#    dataClasLocM='dataclass'
#    dataClasLoc=os.path.join(cwdtop,dataClasLocM)
#    print (dataClasLocM)
#
#    classIncidences=pickle.load( open( os.path.join(dataClasLoc,classNameRefIncidences), "rb" ))
#    classPatho=pickle.load( open( os.path.join(dataClasLoc,classNameRefPatho), "rb" ))
#
#    print ('incidences')
#    for i in range (len(classIncidences)):
#        print (i,findclass(i,classIncidences))
#    print ('patho')
#    for u,v in classPatho.items():
#                print (v,u)
    cwd=os.getcwd()
    print (cv2.__version__)
    print (cwd)
#    keepaenh=1#0 no augmentatiuon, 1 augmentation
##max limit for random value for enhancement 
#    maxmult=0 #multiply pixel value in % => 10 = 1.1
#    maxadd=0 #add data to pixel value in % of max range
#    maxrot=1 #rot + flip max 7 1 for only lrflip
#    maxresize=10 #resize factor in %  ==> 20
#    maxshiftv=5 #shift vertical in percentage of height ==>5 
#    maxshifth=5 #shift horizontal in percentage of width   ==>5
#    maxrotate=10 #degre rotation of image ==>5
    img=cv2.imread('F.png',1)
    img=cv2.resize(img,(100,100),interpolation=cv2.INTER_NEAREST)  
    
    cv2.imshow('orig',img)
    for i in range(8):
        addint,multint,rotimg,resiz,shiftv,shifth,rotateint=generandom(maxadd,
                                    maxmult,maxrot,maxresize,maxshiftv,maxshifth,maxrotate,keepaenh)
        imga=geneaug(img,addint,multint,rotimg,resiz,shiftv,shifth,rotateint,False,img.min(),img.max())

        cv2.imshow('aug'+str(i),imga)
    #        cv2.imwrite(os.path.join(wdir,filename+'.jpg'),b)
    #        cv2.imshow('o',b)
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()