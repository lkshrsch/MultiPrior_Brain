#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
import numpy as np
wd = os.getcwd()

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'fullHeadSegmentation'

############################## Load dataset #############################
     
TPM_channel = ''

train_TPM_channel = ''
    
trainChannels = ['/CV_folds/aphasic_stroke/MRIs_train_set2.txt']
trainLabels = '/CV_folds/aphasic_stroke/labels_train_set2.txt'
    
validation_TPM_channel = ''

validationChannels = ['/CV_folds/aphasic_stroke/MRIs_validation_set2.txt']
validationLabels = '/CV_folds/aphasic_stroke/labels_validation_set2.txt'
    
output_classes = 7
test_subjects = 15
  
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic', 'BIG_multiscale_CNN_TPM', 'BIG_multiscale_CNN_TPM_flexible'
model = 'Baseline' #'MultiPriors' 
dpatch= 57                  # During training       // has to be odd number!
segmentation_dpatch = 51*3  # During segmentation   // has to be odd number!
L2 = 0.00001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice7' #'Dice7'

early_stop_patience = 5

load_model = False
path_to_model = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/training_sessions/MultiPriors_fullHeadSegmentation_configFile_Aphasic_Stroke_2019-08-13_1822/models/fullHeadSegmentation_configFile_Aphasic_Stroke_2019-08-13_1822.log_epoch40.h5'
if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 5e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 1
epochs = 100
samplingMethod_train = 0
data_augmentation = True
proportion_to_flip = 0.5

n_patches = 32*200 
n_subjects = 32 # Check that this is not larger than subjects in training file
size_minibatches = 32 # Check that this value is not larger than the ammount of patches per subject per class

quickmode = False # Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 3*200
n_subjects_val =  3# Check that this is not larger than subjects in validation file
size_minibatches_val = 32 # Check that this value is not larger than the ammount of patches per subject per class
samplingMethod_val = 0

########################################### TEST PARAMETERS
quick_segmentation = True
dice_compare = True
n_fullSegmentations = 3
#list_subjects_fullSegmentation = []
epochs_for_fullSegmentation = np.arange(10,epochs+1,10)
size_test_minibatches = 1
saveSegmentation = True

threshold_EARLY_STOP = 0.01


#penalty_MATRIX = np.array([[ 1, -1, -1, -1,  0,  0],
#                           [-1,  1,  0,  0, -1, -1],
#                           [-1,  0,  1,  0, -1, -1],
#                           [-1,  0,  0,  1,  0, -1],
#                           [ 0, -1, -1,  0,  1,  0],
#                           [ 0, -1, -1, -1,  0,  1]], dtype='float32')

#penalty_MATRIX[penalty_MATRIX < 0 ] = 0

penalty_MATRIX = np.eye(output_classes,output_classes,dtype='float32')

comments = ''

