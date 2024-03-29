# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:05:24 2019

@author: hirsch
"""


from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal, Orthogonal
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape, Permute
from keras.layers.merge import Concatenate
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import print_summary
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
from keras.activations import softmax


def dice_coef(y_true, y_pred):
    smooth = 1e-9
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + smooth)

def Generalised_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

def Generalised_dice_coef_multilabel6(y_true, y_pred, numLabels=6):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice
    
def Generalised_dice_coef_multilabel7(y_true, y_pred, numLabels=7):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

def w_dice_coef(y_true, y_pred, PENALTY):
    smooth = 1e-9
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f) * PENALTY
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def w_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
                                    
    PENALTY = np.array([[ 1,    0],
                        [-1,  1]], dtype='float32')
                          
    #PENALTY[PENALTY < 0] = -5                      
                             
    dice = []
    for index in range(numLabels):
        dice_class = []
        for j in range(numLabels):
            wDice = w_dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,j], PENALTY[index,j])
            dice_class.append(wDice)
        dice.append(K.sum(dice_class)) 
        
    final_dice = K.sum(dice)
    return numLabels - final_dice

def w_dice_coef_multilabel6(y_true, y_pred, numLabels=6):
                                    
    #PENALTY = np.array([   [ 1, -1, -1, -1,  0,  0],
    #                       [-1,  1,  0,  0, -1, -1],
    #                       [-1,  0,  1,  0, -1, -1],
    #                       [-1,  0,  0,  1,  0, -1],
    #                       [ 0, -1, -1,  0,  1,  0],
    #                       [ 0, -1, -1, -1,  0,  1]], dtype='float32')

    PENALTY = np.array([   [ 1, -1, -1, -1,  0,  0],
                           [-1,  1,  0,  0, -1, -1],
                           [-1,  0,  1,  0, -1, -1],
                           [-1,  0,  0,  1, -1, -1],
                           [ 0, -1, -1, -1,  1,  0],
                           [ 0, -1, -1, -1,  0,  1]], dtype='float32')
                          
    PENALTY[PENALTY < 0] = -1                   

    #PENALTY = np.eye(6,6, dtype='float32')
                             
    dice = []
    for index in range(numLabels):
        dice_class = []
        for j in range(numLabels):
            wDice = w_dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,j], PENALTY[index,j])
            dice_class.append(wDice)
        dice.append(K.sum(dice_class)) 
        
    final_dice = K.sum(dice)
    return numLabels - final_dice

def dice_coef_multilabel0(y_true, y_pred):
    index = 0
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel1(y_true, y_pred):
    index = 1
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel2(y_true, y_pred):
    index = 2
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel3(y_true, y_pred):
    index = 3
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel4(y_true, y_pred):
    index = 4
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel5(y_true, y_pred):
    index = 5
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel6(y_true, y_pred):
    index = 6
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice

    

class MultiPriors_noTPM_Model():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [40, 40, 40, 50, 50, 50, 60, 70] #[50, 50, 50, 50, 50, 100, 100, 100]
        self.fc_features = [150,200, output_classes]
        self.d_factor = 3  # downsampling factor = stride in downsampling pathway
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
        #self.w_initializer=w_initializer, # initialization of layer parameters? Needed here?
        #self.w_regularizer=w_regularizer,
        #self.b_initializer=b_initializer, # initialization of bias parameters? Needed here?
        #self.b_regularizer=b_regularizer,
        #self.acti_func=acti_func
    
    def createModel(self):
        '''Creates model architecture
        Input: Data input dimensions, eventually architecture specifications parsed from a config file? (activations, costFunction, hyperparameters (nr layers), dropout....)
        Output: Keras Model'''
    
        #seed = 1337
    
        #mod1      = Input((self.dpatch,self.dpatch,self.dpatch, self.num_channels))
        mod1      = Input((None,None,None, self.num_channels)) # last channel is the individual TPM
        
        #############   Normal pathway   ##################  
        
        # reduces 57 into 9 ( - 48)        
        
        x1        = Cropping3D(cropping = ((16,16),(16,16),(16,16)), input_shape=(None,None,None, self.num_channels))(mod1)

        # 25  , to 9  =  -16
        
        for feature in self.conv_features:  
            x1        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=he_normal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)
            x1        = BatchNormalization()(x1)   

            
        #############   Downsampled pathway   ##################   
        #x2        = MaxPooling3D(pool_size=(self.d_factor,self.d_factor,self.d_factor), padding="same")(mod1)
        
        x2        = AveragePooling3D(pool_size=(self.d_factor,self.d_factor,self.d_factor), padding="same")(mod1)
        
        # Reduces into by 1/3  = 19, then down to 3 : -16        
        
        
        for feature in self.conv_features:    
            x2        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=he_normal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)
            x2        = BatchNormalization()(x2)   
        
        x2        = UpSampling3D(size=(3,3,3))(x2)
        
        #############   Fully connected layers   ################## 
        
        x        = concatenate([x1,x2])


        x        = Conv3D(filters = self.fc_features[0], 
                           kernel_size = (1,1,1), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)

        
        x        = Conv3D(filters = self.fc_features[1], 
                           kernel_size = (1,1,1), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)

#              
#        x        = Conv3D(filters = self.output_classes, 
#                           kernel_size = (1,1,1), 
#                           #kernel_initializer=he_normal(seed=seed),
#                           kernel_initializer=Orthogonal(),
#                           kernel_regularizer=regularizers.l2(self.L2))(x)
#        #x        = BatchNormalization()(x)
       
#        tpm = Input((None,None,None,6))

        #x4        = Cropping3D(cropping = ((24,24),(24,24),(24,24)), input_shape=(None, None, None, self.num_channels))(mod1)
        #x        = concatenate([x,tpm])#,x4])  #  MIXING ONLY CHANNELS + CHANNELS. 
        
        # Skipping this bandfilter and going straigth to the softmax makes everything pointless (no nonlinearity besides softmax), and pushes performance to the floor.
#        x        = Conv3D(filters = self.fc_features[1], 
#                   kernel_size = (1,1,1), 
#                   kernel_initializer=Orthogonal(),
#                   kernel_regularizer=regularizers.l2(self.L2))(x)
#        x        = LeakyReLU()(x)
#        x        = BatchNormalization()(x)
#
#        x        = Conv3D(filters = self.fc_features[1], 
#                   kernel_size = (1,1,1), 
#                   kernel_initializer=Orthogonal(),
#                   kernel_regularizer=regularizers.l2(self.L2))(x)
#        x        = LeakyReLU()(x)


        x        = Conv3D(filters = self.output_classes, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation(softmax)(x)
        
        model     = Model(inputs=[mod1], outputs=x)
        #print_summary(model, positions=[.33, .6, .67,1])
                  
        #rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-8, decay=self.optimizer_decay)
        
        
        if self.loss_function == 'Multinomial':
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3, dice_coef_multilabel4,dice_coef_multilabel5,dice_coef_multilabel6])
        elif self.loss_function == 'Dice2':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1])
        elif self.loss_function == 'Dice6':
            model.compile(loss=dice_coef_multilabel6, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3, dice_coef_multilabel4,dice_coef_multilabel5])
        elif self.loss_function == 'wDice6':
            model.compile(loss=w_dice_coef_multilabel6, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3, dice_coef_multilabel4,dice_coef_multilabel5]) 
        elif self.loss_function == 'wDice2':
            model.compile(loss=w_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1]) 
        elif self.loss_function == 'Dice7':
            model.compile(loss=Generalised_dice_coef_multilabel7, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3, dice_coef_multilabel4,dice_coef_multilabel5,dice_coef_multilabel6 ])
        return model


#dm = MultiPriors_noTPM_Model(7, 1, 0.001, [0], 0.01, 0, 'Dice7' )
#model = dm.createModel()            
#model.summary()  
#from keras.utils import plot_model
#plot_model(model, to_file='/home/hirsch/Documents/projects/brainSegmentation/DeepPriors' +'/multiscale_TPM_noTPM.png', show_shapes=True)    
#
##
#X = np.random.randn(1,57,57,57,1)
#y = np.random.binomial(n=1, p=0.5,size=9**3*7).reshape(1,9,9,9,7)
#y.shape
#
#TPM = np.random.randn(1,9,9,9,6)
#
#yhat = model.predict([X,TPM])
#yhat.shape
#
#model.fit([X,TPM], y)