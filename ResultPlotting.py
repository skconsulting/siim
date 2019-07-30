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


#sys.path.append('/home/sylvain/Documents/horasis/IAsys/pixelmap/python')
import matplotlib.pyplot as plt
#from param_pixelmap import reportDir

#reportDir='unetresnet/oea'
#
#reportDir='unetphoe/new'
#reportDir='unetresnet/wocosine'
#reportDir='unetresnetrial/a'
reportDir1='unetresnet/h/4' #
reportDir='unetresnet/i' #with zero



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
def getData(reportDir):
    pfile=os.path.join(cwdtop,reportDir)
#    nb_epoch= 15 #20 aws 1024 30 for 512
#    minlr=0.01
#    maxlr=1.0
#    learning_rate=0.001
#    def cosine_annealing(x):
#    #    lr = learning_rate
#    #    epochs = nb_epoch
#    #    if x>1:
#    #        coef=1.0*max(min(1.0/(1.5*math.log(x,10)),maxlr),minlr)
#    #    else:
#    #        coef=1.0
#        return learning_rate*(np.cos(np.pi*x/nb_epoch)+1.)/2
#    for x in range (18,22):
#        print (x,cosine_annealing(x))
#      
#        
    print ('path to get csv with train data',pfile)
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
     
    print ('all:', ft)
    
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
    print (filei)
    i=0
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
        epoch=[]
    
    #    print reader
    
        for row in reader:
        
                    categorical_accuracy.append([float(row[acc])])
                    val_accuracy.append(float(row[val_acc]))
            #        lr.append([float(row['lr'])])
                    train_loss.append(float(row[loss]))
                    val_lossd.append(float(row[val_loss]))
                    dice_coefd.append(float(row[dice_coef]))
                    val_dice_coefd.append(float(row[val_dice_coef]))
                    epoch.append(int(row['epoch']))
                    x.append(i)
                    i+=1

                    try:                
                        lrd.append(float(row[lr]))
                    except:
                            pass
#    print('x    ',x)
#    print('epoch',epoch)

    print ( '--------------')
#    print ('Current Last Epoch: ',row['epoch'][0:4])
    print ('Current Last Epoch: ',x[-1],reportDir)

    print ('train_loss',' val_loss', ' train_acc',' val_acc',' dice_coef',' val_dice_coef')
    
    print ( row[loss][0:6],'%12s'%row[val_loss][0:6],'%8s'%row[acc][0:4],
           '%9s'%row[val_acc][0:4],'%8s'%row[dice_coef][0:4],'%10s'%row[val_dice_coef][0:4])
    
    print ('--------------')
#    pic=np.zeros((int(row['epoch'])+1),np.float)
#    pic0=np.zeros((int(row['epoch'])+1),np.float)
#    picloss=np.zeros((int(row['epoch'])+1),np.float)
#    picloss0=np.zeros((int(row['epoch'])+1),np.float)
    pic=np.zeros((len(x)),np.float)
    pic0=np.zeros(len(x),np.float)
    picloss=np.zeros((len(x)),np.float)
    picloss0=np.zeros(len(x),np.float)

    
    limb=0
    #val_l=list(val_lossd)
    val_l=list(val_dice_coefd)
    Reverse=True
    #print val_l
    print ( '-------dice coeff-------')
    #for i in range(int(row['epoch'])):
    try:
        for i in range(3):
        #    try:
            print ('-----')
            print (i,'maximum maximorum for dice_coef',reportDir)
            print ('max val dice_coef starting from epoch:',str(limb),sorted(val_l[limb:],reverse=Reverse)[i])
            print ('epoch for max:',val_l.index(sorted(val_l[limb:],reverse=Reverse)[i]),
                   ' epoch: ',epoch[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])])
            if i==0:
                pic0[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])]=sorted(val_l[limb:],reverse=Reverse)[i]
            else:
                pic[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])]=sorted(val_l[limb:],reverse=Reverse)[i]
            print ('min val loss for max min dice_coef',val_lossd[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])])
    except:
            print ('too few rows')

    
    print ('-----')
    print ('--val loss---')

    val_l=list(val_lossd)
    Reverse=False
    #print val_l
    
    #for i in range(int(row['epoch'])):
    try:
        for i in range(3):
        #    try:
            print ('-----')
            print (i,'minimum minimorum for val loss',reportDir)
            print ('min val loss starting from epoch:',str(limb),sorted(val_l[limb:],reverse=Reverse)[i])
            print ('epoch for min:',val_l.index(sorted(val_l[limb:],reverse=Reverse)[i]))
            if i==0:
                picloss0[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])]=sorted(val_l[limb:],reverse=Reverse)[i]
            else:
                picloss[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])]=sorted(val_l[limb:],reverse=Reverse)[i]
            print ('max valiou for min val loss:',val_dice_coefd[val_l.index(sorted(val_l[limb:],reverse=Reverse)[i])])
    except:
            print ('too few rows')
    #    except:
    #            pass
    return(categorical_accuracy , val_accuracy, train_loss , lrd,  
           train_loss, val_lossd , dice_coefd, val_dice_coefd, x ,pfile,picloss ,picloss0,pic,pic0)
#############################################################################################################################
categorical_accuracy , val_accuracy, train_loss , lrd,  train_loss, val_lossd , dice_coefd, val_dice_coefd, x ,pfile,picloss,picloss0,pic,pic0=getData(reportDir)
categorical_accuracy1 , val_accuracy1, train_loss1 , lrd1,  train_loss1, val_lossd1 , dice_coefd1, val_dice_coefd1, x1 ,pfile1,picloss1,picloss01,pic1,pic01=getData(reportDir1)

#################################################################################################################"
# plotting
##########accuracy
ne=min(len(x),len(x1))
if len(x1)>len(x):
    print('x1 more than x',len(x1),len(x))
#    val_accuracy=val_accuracy+[val_accuracy[-1]]*(len(x1)-len(x))
#    train_loss=train_loss+[train_loss[-1]]*(len(x1)-len(x))
#    val_lossd=val_lossd+[val_lossd[-1]]*(len(x1)-len(x))
#    dice_coefd=dice_coefd+[dice_coefd[-1]]*(len(x1)-len(x))
#    val_dice_coefd=val_dice_coefd+[val_dice_coefd[-1]]*(len(x1)-len(x))
#    val_dice_coefd=val_dice_coefd+[val_dice_coefd[-1]]*(len(x1)-len(x))
#    categorical_accuracy=categorical_accuracy+[categorical_accuracy[-1]]*(len(x1)-len(x))
    categorical_accuracy1=categorical_accuracy1[0:ne]
    val_accuracy1=val_accuracy1[0:ne]

    train_loss1=train_loss1[0:ne]
    val_lossd1=val_lossd1[0:ne]
    dice_coefd1=dice_coefd1[0:ne]
    val_dice_coefd1=val_dice_coefd1[0:ne]



#    for i in range (ne,len(x1)):
#        x.append(str(i))

    xa=x1[0:ne]

else:
    xa=x[0:ne]
    print('x more than x1',len(x1),len(x))
    val_accuracy1=val_accuracy1+[val_accuracy1[-1]]*(len(x)-len(x1))

    train_loss1=train_loss1+[train_loss1[-1]]*(len(x)-len(x1))
    val_lossd1=val_lossd1+[val_lossd1[-1]]*(len(x)-len(x1))
    dice_coefd1=dice_coefd1+[dice_coefd1[-1]]*(len(x)-len(x1))
    val_dice_coefd1=val_dice_coefd1+[val_dice_coefd1[-1]]*(len(x)-len(x1))
    categorical_accuracy1=categorical_accuracy1+[categorical_accuracy1[-1]]*(len(x)-len(x1))


    
print ('last epoch min:',ne)
#print('0',len(val_accuracy))
#print('1',len(val_accuracy1))
#print(len(x))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Accuracy '+reportDir+' as master',fontsize=10)
ax.set_xlabel('# Epochs')
ax.set_ylabel('value')
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('both')



#plt.ylim(0.95,1.0)
#plt.ylim(0.85,0.96)
ax.plot(x,categorical_accuracy, label='categorical_accuracy');
ax.plot(x,categorical_accuracy1, label='categorical_accuracy1');

ax.plot(x,val_accuracy, label='val_categorical_accuracy');
ax.plot(x,val_accuracy1, label='val_categorical_accuracy1');


#ax.plot(x,train_loss, label='train_loss');
#ax.plot(x,val_loss, label='val_loss');

legend = ax.legend(loc='lower right', shadow=True,fontsize=10)
plt.savefig(os.path.join(pfile,'acc.png'))
plt.show()

#####loss

fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.set_title('Loss '+reportDir+' as master',fontsize=10)

#ax.set_ylim(0.,0.06)
limb=0
limbmax=4
limbmax=len(x)
#limbmax=ne

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.plot(x[limb:limbmax],train_loss[0:limbmax], label='loss');
ax.plot(x[limb:limbmax],train_loss1[limb:limbmax], label='loss1');
#
ax.plot(x[limb:limbmax],val_lossd[limb:limbmax], label='val_loss');
ax.plot(x[limb:limbmax],val_lossd1[limb:limbmax], label='val_loss1');

print(len(x[limb:limbmax]))
print(len(picloss[limb:limbmax]))
ax.plot(x[limb:limbmax],picloss[limb:limbmax], label='val_loss pic');
ax.plot(x[limb:limbmax],picloss0[limb:limbmax], label='val_loss pic min');
#ax1.set_ylim(0.,1e-1)
if len(lrd)>0:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax1=ax.twinx()
    ax1.plot(x[limb:limbmax],lrd[limb:limbmax], label='lr',color='violet');
    ax1.set_ylabel('lr', color='violet')
    ax1.tick_params('y', colors='violet')

legend = ax.legend(loc='lower left', shadow=False,fontsize=10)
plt.savefig(os.path.join(pfile,'loss.png'))
plt.show()

######### iou

fig = plt.figure()
#plt.ylim(0.45,0.55)
ax = fig.add_subplot(1,1,1)
ax.set_title('Dice coeff '+reportDir+' as master',fontsize=10)
ax.plot(x,dice_coefd, label='dice_coef');
ax.plot(x,dice_coefd1, label='dice_coef1');
ax.plot(x,val_dice_coefd, label='val_dice_coef');
ax.plot(x,val_dice_coefd1, label='val_dice_coef1');
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
if len(lrd)>0:
    ax1=ax.twinx()
    ax1.plot(x,lrd, label='lr',color='violet');
    ax1.set_ylabel('lr', color='violet')
    ax1.tick_params('y', colors='violet')

ax.plot(x,pic, label='val_dice_coef pic');
ax.plot(x,pic0, label='val_dice_coef pic max');
legend = ax.legend(loc='lower left', shadow=True,fontsize=10)
plt.savefig(os.path.join(pfile,'iou.png'))
plt.show()
