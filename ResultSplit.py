# coding: utf-8
"""
# # Plot Results of training
# allows interpretation of impact of various parameters setting 
# 
# ### needs correct <file>.csv file

"""
import math
import numpy as np
import csv
import os
import sys

nbSplit=10
max_dice_coef=0
max_dice_coef_i=100
min_val_loss=100
min_val_loss_i=0
epoch_for_max_dice_coef=0
epoch_for_min_val_loss=0

#sys.path.append('/home/sylvain/Documents/horasis/IAsys/pixelmap/python')
import matplotlib.pyplot as plt
#from param_pixelmap import reportDir

#reportDir='unetresnet/oea'
#
#reportDir='unetphoe/new'
#reportDir='unetresnet/wocosine'
#reportDir='unetresnetrial/a'
reportDir='unetresnet/j' #
#reportDir='UResNet34/a' #
#reportDir='UResNet34trial/a' #


#reportDir='r0ncD0/a'




cwd=os.getcwd()

#train directory: trainset/setname/weights
(cwdtop,tail)=os.path.split(cwd)
#p1='trainset'
#p2='set1' #extensioon for path for data for training
#p3='report'
#reportDir='ref2-unet'
#p3='report'
#reportDir='r0nc'

#pfile=cwd

##########################################################################
def getData(nS):
    global max_dice_coef,max_dice_coef_i,min_val_loss,min_val_loss_i,epoch_for_max_dice_coef,epoch_for_min_val_loss
    pfile=os.path.join(cwdtop,reportDir,nS)
#    print ('path to get csv with train data',pfile)
    # which file is the source
    fileis = [name for name in os.listdir(pfile) if "e.csv" in name.lower()]
    #print filei
#    ordfb=[]
    ordf=[]
    for f in fileis:
       nbs = os.path.getmtime(os.path.join(pfile,f))
       tt=(f,nbs)
       ordf.append(tt)
    #ordlistfb=sorted(ordfb,key=lambda col:col[1],reverse=True)
    #print ordlistfb
    #fb=ordlistfb[0][0]
    ordlistft=sorted(ordf,key=lambda col:col[1],reverse=True)
    #print ordlistft
    ft=ordlistft[0][0]
     
#    print ('all:', ft)
    
    acc='acc'
    val_acc='val_acc'
    #acc='categorical_accuracy'
    #val_acc='val_categorical_accuracy'
    loss='loss'
    val_loss='val_loss'
    lr='lr'
    dice_coef='dice_coef'
    val_dice_coef='val_dice_coef'
    #my_iou_metric='my_iou_metric'
    #val_my_iou_metric='val_my_iou_metric'
    
    
    filei = os.path.join(pfile,ft)
#    print (filei)
    with open(filei, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        categorical_accuracy = []
        val_accuracy = []
        train_loss = []
        lrd=[]
        train_loss = []
        val_lossd = []
        dice_coefd=[]
        val_dice_coefd=[]
        x = []
    
    #    print reader
    
        for row in reader:
        
                    categorical_accuracy.append([float(row[acc])])
                    val_accuracy.append(float(row[val_acc]))
            #        lr.append([float(row['lr'])])
                    train_loss.append(float(row[loss]))
                    val_lossd.append(float(row[val_loss]))
                    dice_coefd.append(float(row[dice_coef]))
                    val_dice_coefd.append(float(row[val_dice_coef]))
                    x.append(int(row['epoch']))
                    try:                
                        lrd.append(float(row[lr]))
                    except:
                            pass
    
#    print ( '--------------')
#    print ('Current Last Epoch: ',row['epoch'][0:4])
#    print ('train_loss',' val_loss', ' train_acc',' val_acc',' dice_coef',' val_dice_coef')
#    
#    print ( row[loss][0:6],'%12s'%row[val_loss][0:6],'%8s'%row[acc][0:4],
#           '%9s'%row[val_acc][0:4],'%8s'%row[dice_coef][0:4],'%10s'%row[val_dice_coef][0:4])
#    
#    print ('--------------')
    pic=np.zeros((int(row['epoch'])+1),np.float)
    pic0=np.zeros((int(row['epoch'])+1),np.float)
    picloss=np.zeros((int(row['epoch'])+1),np.float)
    picloss0=np.zeros((int(row['epoch'])+1),np.float)
    
    
    limb=0
    #val_l=list(val_lossd)
    val_l=list(val_dice_coefd)
    Reverse=True
    #print val_l
    print ( '-------dice coeff-------',reportDir,nS)
    #for i in range(int(row['epoch'])):
    
#    print ('-----')
#    print ('maximum maximorum for dice_coef',reportDir,nS)
    print ('max val dice_coef starting from epoch:',str(limb),sorted(val_l[limb:],reverse=Reverse)[0])
    if sorted(val_l[limb:],reverse=Reverse)[0]>max_dice_coef:
        max_dice_coef=sorted(val_l[limb:],reverse=Reverse)[0]
        max_dice_coef_i=nS
        epoch_for_max_dice_coef=val_l.index(sorted(val_l[limb:],reverse=Reverse)[0])
        
    print ('epoch for max:',val_l.index(sorted(val_l[limb:],reverse=Reverse)[0]))
    
#    print ('-----')
    print ('--val loss---',reportDir,nS)

    val_l=list(val_lossd)
    Reverse=False

#    print ('-----')
#    print ('minimum minimorum for val loss')
    print ('min val loss starting from epoch:',str(limb),sorted(val_l[limb:],reverse=Reverse)[0])
    print ('epoch for min:',val_l.index(sorted(val_l[limb:],reverse=Reverse)[0]))
    if sorted(val_l[limb:],reverse=Reverse)[0]<min_val_loss:
        min_val_loss=sorted(val_l[limb:],reverse=Reverse)[0]
        min_val_loss_i=nS
        epoch_for_min_val_loss=val_l.index(sorted(val_l[limb:],reverse=Reverse)[0])

    print ('-----')

    #            pass
    return(categorical_accuracy , val_accuracy, train_loss , lrd,  
           train_loss, val_lossd , dice_coefd, val_dice_coefd, x ,pfile,picloss ,picloss0,pic,pic0)
#############################################################################################################################

for ns in range(nbSplit):
    categorical_accuracy , val_accuracy, train_loss , lrd,  train_loss, val_lossd , dice_coefd, val_dice_coefd, x ,pfile,picloss,picloss0,pic,pic0=getData(str(ns))
print('number split max dice_coef: ',max_dice_coef_i,' max dice coef: ', max_dice_coef, 'epoch: ',epoch_for_max_dice_coef)
print('number split min val_loss: ',min_val_loss_i,' min val loss: ', min_val_loss,' epoch: ',epoch_for_min_val_loss)

