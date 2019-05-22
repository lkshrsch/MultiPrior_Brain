#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'fullHeadSegmentation'

############################## Load dataset #############################
 
#TPM_channel = '/home/hirsch/Documents/projects/TPM/correct_labels_TPM_padded.nii'
    
TPM_channel = '/logTPM.nii'
    
trainChannels = ['/CV_folds/stroke/MRIs_train_set0.txt']
trainLabels = '/CV_folds/stroke/labels_train_set0.txt'
    
testChannels = ['/CV_folds/stroke/MRIs_validation_set0.txt']
testLabels = '/CV_folds/stroke/labels_validation_set0.txt'

validationChannels = testChannels
validationLabels = testLabels
    
output_classes = 7
test_subjects = 15
  
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic', 'BIG_multiscale_CNN_TPM', 'BIG_multiscale_CNN_TPM_flexible', 'DeepMedic_TPM'
model = 'DeepMedic_TPM'
dpatch= 57                  # During training
segmentation_dpatch = 57*3  # During segmentation
L2 = 0.00001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice7' #'Dice7'

early_stop_patience = 4

load_model = False
path_to_model = '/home/andy/projects/brain_segmentation/training_sessions/DeepMedic_TPM_fullHeadSegmentation_configFile0_DeepMedic_TPM_2019-02-13_1624/models/M_fullHeadSegmentation_configFile0_DeepMedic_TPM_2019-02-13_1624.log_epoch1.h5'
if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 5e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 5
epochs = 30
samplingMethod_train = 0

n_patches = 6000#12650
n_subjects = 33#42# Check that this is not larger than subjects in training file
size_minibatches = 64 # Check that this value is not larger than the ammount of patches per subject per class

quickmode = True # Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 500
n_subjects_val =  1# Check that this is not larger than subjects in validation file
size_minibatches_val = 64 # Check that this value is not larger than the ammount of patches per subject per class
samplingMethod_val = 0

########################################### TEST PARAMETERS
quick_segmentation = True
n_fullSegmentations = 14
#list_subjects_fullSegmentation = []
epochs_for_fullSegmentation = range(1,epochs)
size_test_minibatches = 1
saveSegmentation = False

threshold_EARLY_STOP = 0.001

import numpy as np
#penalty_MATRIX = np.array([[ 1, -1, -1, -1,  0,  0],
#                           [-1,  1,  0,  0, -1, -1],
#                           [-1,  0,  1,  0, -1, -1],
#                           [-1,  0,  0,  1,  0, -1],
#                           [ 0, -1, -1,  0,  1,  0],
#                           [ 0, -1, -1, -1,  0,  1]], dtype='float32')

#penalty_MATRIX[penalty_MATRIX < 0 ] = 0

penalty_MATRIX = np.eye(output_classes,output_classes,dtype='float32')

comments = ''

