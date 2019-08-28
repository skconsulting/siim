# -*- coding: utf-8 -*-
"""
Created on April 9th, 2018
list models for segmentation

@author: sylvain
"""

#from param_seg import modelName
import pickle

import numpy as np

import sys
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,UpSampling2D,Add,Concatenate
import keras
from keras import backend as K
from Uresnet34 import UResNet34
from unetplusplus import UEfficientNet
from keras.losses import binary_crossentropy
import tensorflow as tf

print (' keras.backend.image_data_format :',keras.backend.image_data_format())
##############"
#  channel last with tf
############
DIM_ORDERING=keras.backend.image_data_format()


#loss functions
#def dice_coef(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred = K.cast(y_pred, 'float32')
#    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
#    intersection = y_true_f * y_pred_f
#    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
#    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch

def generalized_dice_loss_w(y_true, y_pred): 
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    Ncl = y_pred.shape[-1]
    w = np.zeros((Ncl,))
    for l in range(0,Ncl): w[l] = np.sum( np.asarray(y_true[:,:,:,l]==1,np.int8) )
    w = 1/(w**2+0.00001)

    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2))
    numerator = K.sum(numerator)
    
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2))
    denominator = K.sum(denominator)
    
    gen_dice_coef = numerator/denominator
    
    return 1-2*gen_dice_coef

def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)

    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_coeff_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

#def dice_coef(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

#
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')

    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefX(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score
#

def bce_dice_lossr34(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def dice_coefN(y_true, y_pred,smooth=1.):
    y_true_f = keras.layers.Flatten()(y_true)
    y_pred_f = keras.layers.Flatten()(y_pred)
    intersection = keras.layers.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.layers.reduce_sum(y_true_f) + keras.layers.reduce_sum(y_pred_f) + smooth)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return 1.0 -dice_coef(y_true, y_pred)

def generalized_dice_w(y_true, y_pred): 
    return -generalized_dice_loss_w(y_true, y_pred)


def balanced_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    return tf.reduce_mean(loss * (1 - beta))

  return loss


def weightedLoss(originalLossFunc, weightsList):
    def lossFunc(true, pred):
        axis = -1 #if channels last 
        #axis=  1 #if channels first

        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.cast(K.argmax(true, axis=axis),"int32")

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   
#        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]


        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss    


def iou_loss(y_true, y_pred):
#    y_true =  K.reshape(y_true, [-1])
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)




#    y_pred =  K.reshape(y_pred, [-1])
    intersection =  tf.reduce_sum(y_true_f * y_pred_f)
    score = (intersection + 1.) / ( tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + 1.)
    return 1.0 - score



def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true)*alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1-p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss    




def binary_focal_loss0(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed



# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5* keras.losses.binary_crossentropy(y_true, y_pred) + 0.5*iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred =  round(y_pred)
    intersect =  K.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union =  K.reduce_sum(y_true, axis=[1, 2, 3]) +  K.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth =  K.ones( K.shape(intersect))
    return  K.reduce_mean((intersect + smooth) / (union - intersect + smooth))


def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

#####################################################
#models
#####################################################
def build_model_unetphoe(input_layer, start_neurons):
        #ref: https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification
        # 128 -> 64
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)
    
        # 64 -> 32
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)
    
        # 32 -> 16
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)
    
        # 16 -> 8
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)
    
        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
        # 8 -> 16
        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    
        # 16 -> 32
        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    
        # 32 -> 64
        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    
        # 64 -> 128
        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
        #uconv1 = Dropout(0.5)(uconv1)
        output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
        
        return output_layer

def unetphoe(num_classl,INP_SHAPEl,DIM_ORDERINGl,CONCAT_AXISl,d0):
#        input_layer = Input((img_rows, img_cols, 1))
        input_layer = Input(shape=INP_SHAPEl)
        output_layer = build_model_unetphoe(input_layer, 16)
        
        
        # In[ ]:
        
        
        model = Model(input_layer, output_layer)
        return model


###############      downsample_resblock
def create_downsample(channels, inputs):
    x =  keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x =  keras.layers.LeakyReLU(0)(x)
    x =  keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x =  keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs,DropoutRatio):
    x =  keras.layers.LeakyReLU(0)(inputs)
    x =  keras.layers.BatchNormalization(momentum=0.9)(x)
    x =  keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x =  keras.layers.LeakyReLU(0)(x)
    x =  keras.layers.BatchNormalization(momentum=0.9)(x)
    x =  keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    if DropoutRatio>0:
        x =  keras.layers.Dropout(DropoutRatio)(x)
    return  keras.layers.add([x, inputs])

def downsample_resblock(num_classl,INP_SHAPEl,DIM_ORDERINGl,CONCAT_AXISl,d0):

    print ('downsample_resblock with number of classes:',num_classl ,' and d0:', d0)
    n_blocks=2
    depth=4

#    d0=0
    channels=32
    inputs =  keras.Input(shape=INP_SHAPEl)

    x =  keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x,d0)
    # output
    x =  keras.layers.LeakyReLU(0)(x)
    x =  keras.layers.BatchNormalization(momentum=0.9)(x)
    x =  keras.layers.Conv2D(256, 1, activation=None)(x)
    x =  keras.layers.LeakyReLU(0)(x)
    x =  keras.layers.BatchNormalization(momentum=0.9)(x)   
    x =  keras.layers.Conv2DTranspose(128, (8,8), strides=(4,4), padding="same", activation=None)(x)
    x =  keras.layers.LeakyReLU(0)(x)
    x =  keras.layers.BatchNormalization(momentum=0.9)(x)
    x =  keras.layers.Conv2D(num_classl, 1, activation='sigmoid')(x)
    
    outputs =  keras.layers.UpSampling2D(2**(depth-2))(x)
    model =  keras.Model(inputs=inputs, outputs=outputs)
    return model


#####################################################
###############      unet

def double_conv_layerunet(x, size, dropout, batch_norm,dim_org,ke_i,kernel_siz,padding):
    
    axis = -1 # -1 for channel last
    x = keras.layers.Conv2D(size,kernel_siz, activation=None,kernel_initializer=ke_i,
                  data_format=dim_org,kernel_constraint=maxnorm(4.),padding=padding)(x)  
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(size,kernel_siz, activation=None,kernel_initializer=ke_i,
                  data_format=dim_org,kernel_constraint=maxnorm(4.),padding=padding)(x)  
    x = keras.layers.Activation('relu')(x)
    if batch_norm == True:
        x = keras.layers.BatchNormalization(momentum=0.9,axis=axis)(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    
    return x

def unet(num_class_,INP_SHAPE_,dim_org_,CONCAT_AXIS_,d0):

    print ('this is model UNET new2')
    ke_i='he_normal'
#    ke_i='glorot_uniform'
    kernel_size=(3,3)
    kernel_size1=(2,2)
    kernel_size1=(3,3)
    stride=(2,2)
    coefcon={}
    coefcon[1]=32 #32 for 320

#coefcon[1]=16 #32 for 320 #16 for gpu
    for i in range (2,6):
        coefcon[i]=coefcon[i-1]*2
    print (coefcon)
    dor={}
#    dor[1]=0.04 #0.04 f
    dor[1]=d0 #0.04 f
    
#    dor[1]=0 #0.04 f #best

    for i in range (2,6):
        dor[i]=min(dor[i-1]*2,0.5)
#        dor[i]=0
    print ('do coeff :',dor)
    batch_norm=True
    print ('batchnorm :', batch_norm)
#    padding='valid'
    padding='same' #????
   
    inputs = keras.Input(shape=INP_SHAPE_)

    conv1=double_conv_layerunet(inputs, coefcon[1], dor[1], False,dim_org_,ke_i,kernel_size,padding)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2),data_format=dim_org_)(conv1)

    conv2=double_conv_layerunet(pool1, coefcon[2], dor[2], batch_norm,dim_org_,ke_i,kernel_size,padding)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2),data_format=dim_org_)(conv2)

    conv3=double_conv_layerunet(pool2, coefcon[3], dor[3], batch_norm,dim_org_,ke_i,kernel_size,padding)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2),data_format=dim_org_)(conv3)

    conv4=double_conv_layerunet(pool3, coefcon[4],dor[4], batch_norm,dim_org_,ke_i,kernel_size,padding)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2),data_format=dim_org_)(conv4)

    conv5=double_conv_layerunet(pool4, coefcon[5], dor[5], batch_norm,dim_org_,ke_i,kernel_size,padding)
    
    up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(coefcon[4], kernel_size1, strides=(2, 2), padding=padding,
                                       data_format=dim_org_)(conv5), conv4], axis=CONCAT_AXIS_) 
    conv6=double_conv_layerunet(up6, coefcon[4], dor[4], batch_norm,dim_org_,ke_i,kernel_size,padding)

    up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(coefcon[3], kernel_size1, strides=stride, padding=padding,
                                       data_format=dim_org_)(conv6), conv3], axis=CONCAT_AXIS_)    
    conv7=double_conv_layerunet(up7, coefcon[3], dor[3], batch_norm,dim_org_,ke_i,kernel_size,padding)

    up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(coefcon[2], kernel_size1, strides=stride, padding=padding,
                                       data_format=dim_org_)(conv7), conv2], axis=CONCAT_AXIS_)  
    conv8=double_conv_layerunet(up8, coefcon[2], dor[2], batch_norm,dim_org_,ke_i,kernel_size,padding)

    up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(coefcon[1], kernel_size, strides=stride, padding=padding,
                                       data_format=dim_org_)(conv8), conv1], axis=CONCAT_AXIS_)  
    conv9=double_conv_layerunet(up9, coefcon[1], dor[1], False,dim_org_,ke_i,kernel_size,padding)    
     
    conv10 = keras.layers.Conv2D(int(num_class_), (1,1), activation='sigmoid',

                    data_format=dim_org_,padding=padding,kernel_initializer=ke_i)(conv9) #softmax?  
    model = keras.Model(inputs, conv10)

    
    return model

###################################################################"
#RESUNET
def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output

def residual_block0(x, filters, kernel_size=3, padding='same', strides=1):
    k_size=3
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet(num_class_,INP_SHAPE_,dim_org_,CONCAT_AXIS_,d0):
    f = [16, 32, 64, 128, 256]
#    inputs = Input((img_size, img_size, 1))
    inputs = keras.Input(shape=INP_SHAPE_)
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block0(e1, f[1], strides=2)
    e3 = residual_block0(e2, f[2], strides=2)
    e4 = residual_block0(e3, f[3], strides=2)
    e5 = residual_block0(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block0(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block0(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block0(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block0(u4, f[1])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    return model
#################"
# unetresnet

def BatchActivate(x):
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.BatchNormalization()(x)
#    x = keras.layers.Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = keras.layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = True,k=(3,3)):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, k )
    x = convolution_block(x, num_filters, k, activation=False)
    x = keras.layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def build_model_unetresnet(input_layer, start_neurons, num_class_, DropoutRatio = 0.5,):
    ks0=(3,3)
#    ks1=(5,5)
#    ks2=(7,7)
    # 101 -> 50
    conv1 = keras.layers.Conv2D(start_neurons * 1, ks0, activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1,k=ks0)
    conv1 = residual_block(conv1,start_neurons * 1, True,k=ks0)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)
    pool1 = keras.layers.Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = keras.layers.Conv2D(start_neurons * 2, ks0, activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    pool2 = keras.layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = keras.layers.MaxPooling2D((2, 2))(conv3)
    pool3 = keras.layers.Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = keras.layers.Dropout(DropoutRatio)(pool4)

    # Middle
    convm = keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = keras.layers.concatenate([deconv4, conv4])
    uconv4 = keras.layers.Dropout(DropoutRatio)(uconv4)
    
    uconv4 = keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    deconv3 = keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)

    uconv3 = keras.layers.concatenate([deconv3, conv3])    
    uconv3 = keras.layers.Dropout(DropoutRatio)(uconv3)
    
    uconv3 = keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = keras.layers.concatenate([deconv2, conv2])
        
    uconv2 = keras.layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    deconv1 = keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = keras.layers.concatenate([deconv1, conv1])
    
    uconv1 = keras.layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    output_layer_noActi = keras.layers.Conv2D(num_class_, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  keras.layers.Activation('sigmoid')(output_layer_noActi)
    
    return output_layer


def unetresnet(num_class_,INP_SHAPE_,dim_org_,CONCAT_AXIS_,d0):

    print ('this is model unetresnet with d0: ',d0)
    input_layer = keras.Input(INP_SHAPE_)
#    output_layer = build_model(input_layer, 32, num_class_,0.3)
#    output_layer = build_model_unetresnet(input_layer,start neuron 32 32, num_class_,d0)
    output_layer = build_model_unetresnet(input_layer, 16, num_class_,d0)

    
    model = keras.Model(input_layer, output_layer)
#    model = keras.Model(inputs=[input_layer], outputs=[output_layer])
    return model

def  unetsimple(num_class_,INP_SHAPE_,dim_org_,CONCAT_AXIS_,d0):
    inputs = Input(INP_SHAPE_)
#    inputs = Input((None, None, 1))
    

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    
    c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
    c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (c55)
    
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
    
    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation='relu', padding='same') (u71)
    c61 = Conv2D(32, (3, 3), activation='relu', padding='same') (c71)
    
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
    
    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def  unetsimple0(num_class_,INP_SHAPE_,dim_org_,CONCAT_AXIS_,d0):
    inputs = keras.Input(INP_SHAPE_)
#    inputs = Input((None, None, 1))
    

    c1 =  keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 =  keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 =  keras.layers.MaxPooling2D((2, 2)) (c1)
    
    c2 =  keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 =  keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 =  keras.layers.MaxPooling2D((2, 2)) (c2)
    
    c3 =  keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 =  keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 =  keras.layers.MaxPooling2D((2, 2)) (c3)
    
    c4 =  keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 =  keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 =  keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 =  keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
    c5 =  keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
    p5 =  keras.layers.MaxPooling2D(pool_size=(2, 2)) (c5)
    
    c55 =  keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
    c55 =  keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c55)
    
    u6 =  keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
    u6 =  keras.layers.concatenate([u6, c5])
    c6 =  keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 =  keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
    
    u71 =  keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u71 =  keras.layers.concatenate([u71, c4])
    c71 =  keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u71)
    c61 =  keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c71)
    
    u7 =  keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)
    u7 =  keras.layers.concatenate([u7, c3])
    c7 =  keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 =  keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
    
    u8 =  keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 =  keras.layers.concatenate([u8, c2])
    c8 =  keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 =  keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
    
    u9 =  keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 =  keras.layers.concatenate([u9, c1], axis=3)
    c9 =  keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 =  keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
    
    outputs =  keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

##########################################################################
############################################################
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def get_model(modelName_,num_class_,num_bit_,image_rows_,image_cols_,mat_t_k_,
                                      weights_,weightedl_,namelastc_,learning_rate_,comPile_,d0=0.5):
    DIM_ORDERING=keras.backend.image_data_format()
#    print DIM_ORDERING
    if DIM_ORDERING == 'channels_first':
        INP_SHAPE = (num_bit_, image_rows_, image_cols_)  
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'channels_last':
        INP_SHAPE = (image_rows_, image_cols_, num_bit_)  
        CONCAT_AXIS = 3

    if modelName_ == 'unet':
        model = unet(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)

    elif modelName_ == 'downsample_resblock':   
        model = downsample_resblock(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
        
    elif modelName_ == 'unetresnet':   
        model = unetresnet(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
        
    elif modelName_ == 'unetsimple':   
        model = unetsimple(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
    elif modelName_ == 'unetphoe':   
        model = unetphoe(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
    elif modelName_ == 'ResUNet':   
        model = ResUNet(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
    elif modelName_ == 'UResNet34':   
        INP_SHAPE = (num_bit_, image_rows_, image_cols_) 
        model = UResNet34(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
    elif modelName_ == 'UEfficientNet':    
#        model = UEfficientNet(num_class_,INP_SHAPE,DIM_ORDERING,CONCAT_AXIS,d0)
        model = UEfficientNet(input_shape=INP_SHAPE,dropout_rate=d0)

    else:
            print ('not defined model')
            sys.exit()
    if namelastc_ != 'NAN':
          print ('load weight ',namelastc_)
          model.load_weights(namelastc_) 

    if comPile_:
        if weightedl_:
            print ('weighted loss')
#            opt =  keras.optimizers.Adam(lr=learning_rate_)
            opt = Adam(lr=learning_rate_)

#            mloss = weighted_categorical_crossentropy(weights_)
            mloss = weightedLoss(dice_coef_loss,weights_)

#            model.compile(optimizer=opt, loss=mloss, metrics=['categorical_accuracy'])
            model.compile(optimizer=opt, loss=mloss, metrics=["accuracy",dice_coef])
        else:
            print ('NO weighted loss')
            if modelName_=='UEfficientNet': 
                opt = Adam(lr=learning_rate_)
#                opt = SGD(lr=learning_rate_, momentum=0.9)

                lr_metric = get_lr_metric(opt)
                model.compile(loss=bce_dice_loss, optimizer=opt, metrics=["accuracy",my_iou_metric,lr_metric,dice_coef])
#                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['categorical_accuracy',dice_coef]) # to be used with cosine annealing
#                model.compile(optimizer=opt, loss=iou_bce_loss, metrics=['accuracy', dice_coef,lr_metric]) # to be used with cosine annealing
#                model.compile(loss= [dice_loss] , optimizer=opt, metrics=["accuracy",dice_coef])
#                model.compile(loss=[binary_focal_loss(alpha=.25,gamma=2)], optimizer=opt, metrics=["accuracy",dice_coef,lr_metric])
                
                
#                model.compile(loss=dice_coef_loss, optimizer=opt, metrics=["accuracy",dice_coef, lr_metric]) # recommended 
            elif modelName_=='unetresnet': 
                opt = Adam(lr=learning_rate_)
#                opt = SGD(lr=learning_rate_, momentum=0.9)

                lr_metric = get_lr_metric(opt)
#                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['categorical_accuracy',dice_coef]) # to be used with cosine annealing
#                model.compile(optimizer=opt, loss=iou_bce_loss, metrics=['accuracy', dice_coef,lr_metric]) # to be used with cosine annealing
#                model.compile(loss= [dice_loss] , optimizer=opt, metrics=["accuracy",dice_coef])
#                model.compile(loss=[binary_focal_loss(alpha=.25,gamma=2)], optimizer=opt, metrics=["accuracy",dice_coef,lr_metric])
                
                
                model.compile(loss=dice_coef_loss, optimizer=opt, metrics=["accuracy",dice_coef, lr_metric]) # recommended 
                
            elif modelName_=='UResNet34': 
#                opt = Adam(lr=learning_rate_)
                opt = SGD(lr=learning_rate_, momentum=0.9)

                lr_metric = get_lr_metric(opt)
#                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['categorical_accuracy',dice_coef]) # to be used with cosine annealing
#                model.compile(optimizer=opt, loss=bce_dice_loss, metrics=['accuracy', my_iou_metric]) # to be used with cosine annealing
#                model.compile(loss= [dice_loss] , optimizer=opt, metrics=["accuracy",dice_coef])
#                model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",dice_coef])
#                model.compile(loss=bce_dice_lossr34, optimizer=opt, metrics=["accuracy",dice_coef, lr_metric])
                model.compile(loss=dice_coef_loss, optimizer=opt, metrics=["accuracy",dice_coef, lr_metric])


            elif modelName_=='downsample_resblock': 
                opt =  keras.optimizers.Adam(lr=learning_rate_)
                lr_metric = get_lr_metric(opt)
#                model.compile(optimizer=opt, loss=bce_dice_loss, metrics=['accuracy', my_iou_metric]) # to be used with cosine annealing
                model.compile(loss=dice_coef_loss, optimizer=opt, metrics=["accuracy",dice_coef, lr_metric])

            elif modelName_=='unet': 
                opt =  keras.optimizers.Adam(lr=learning_rate_)
#                model.compile(optimizer=opt, loss=bce_dice_loss, metrics=['accuracy', my_iou_metric]) # to be used with cosine annealing
                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',dice_coef]) # to be used with cosine annealing

            elif modelName_=='unetsimple': 
                opt =  keras.optimizers.Adam(lr=learning_rate_)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef]) # to be used with cosine annealing
            elif modelName_=='unetphoe': 
                opt = Adam(lr=learning_rate_)

                model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",dice_coef])
#                model.compile(loss=[dice_coef_loss], optimizer=opt, metrics=["accuracy",dice_coef])


#                model.compile(loss= [dice_loss], optimizer=opt, metrics=["accuracy",dice_coef])

#
#                opt =  keras.optimizers.Adam(lr=learning_rate_)
#                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef]) # to be used with cosine annealing
           
            
            
            elif modelName_=='ResUNet': 
                opt = Adam(lr=learning_rate_)

                model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",dice_coef])
            else:
                print ('not defined model')
                sys.exit()
      
#    for i, layer in enumerate(model.layers):
#        print(i, layer.name)

    print ('last model layer',model.layers[-1].output_shape) #== (None, 16, 16, 21)
    return model

if __name__ == "__main__":
   modelNamei='unetresnet'
#   modelNamei='downsample_resblock'
#   modelNamei='unetsimple'
#   modelNamei='ResUNet'
#   modelNamei='UResNet34'
   modelNamei='UEfficientNet'
   weights=[0.1, 0.9]
#   weights = np.array([0.1,2])
   num_classi=1
   weightedl=False
   num_biti=3
   img_rowsi=128
   img_colsi=128
   ns=0
   learning_rate=1e-3
   print(modelNamei)

   model=get_model(modelNamei,num_classi,num_biti,img_rowsi+ns,img_colsi+ns,True,
                   weights,weightedl,'NAN',learning_rate,True,0.04)
#   model.summary()

#   for i, layer in enumerate(model.layers):
#        print(i, layer.name)
   DIM_ORDERING=keras.backend.image_data_format()
   print (DIM_ORDERING)
   
   if DIM_ORDERING == 'channels_first':
        imarr = np.ones((num_biti,img_rowsi,img_colsi))
   else:
        imarr = np.ones((img_rowsi,img_colsi,num_biti))
   imarr = np.expand_dims(imarr, axis=0)
   print ('imarr.shape',imarr.shape)
   print ('model.predict(imarr).shape ',model.predict(imarr,verbose=1).shape)
   
   json_string = model.to_json()
   pickle.dump(json_string,open( modelNamei+'CNN.h5', "wb"),protocol=-1)
   
   print (model.metrics_names)
   
#   model_json=model.to_json()
#   with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#   file=open('model8s.json','w')
#   
#   file.write(mjson)
#   file.close()
   
#   
   orig_stdout = sys.stdout
   f = open(modelNamei+'_model.txt', 'w')
   sys.stdout = f
   print(model.summary())
   sys.stdout = orig_stdout
   f.close()