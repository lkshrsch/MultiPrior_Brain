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

dataset = 'myDataSet'

############################## Load dataset #############################
     
TPM_channel = ''

train_TPM_channel = ''
    
trainChannels = ['/CV_folds/Training_Data.txt']
trainLabels = '/CV_folds/Training_Labels.txt'
    
validation_TPM_channel = ''

validationChannels = ['/CV_folds/Validation_Data.txt']
validationLabels = '/CV_folds/Validation_Labels.txt'
    
output_classes = 7
test_subjects = 15  # rename to validation_subjects
  

######################################### MODEL PARAMETERS
model = 'Baseline' 
dpatch= 57                  # During training       // has to be odd number!
segmentation_dpatch = 51*3  # During segmentation   // has to be odd number!
L2 = 0.00001
loss_function = 'Dice7' 

load_model = False
path_to_model = ''

dropout = [0,0]  
learning_rate = 5e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 1
epochs = 100
samplingMethod_train = 0
data_augmentation = True
proportion_to_flip = 0.5
early_stop_patience = 5

n_patches = 32*200 
n_subjects = 32 	
size_minibatches = 32 	

quickmode = False 	# Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 3*200
n_subjects_val =  3
size_minibatches_val = 32 
samplingMethod_val = 0

########################################### TEST PARAMETERS
quick_segmentation = True
dice_compare = True
n_fullSegmentations = 3
softmax_output = False
#list_subjects_fullSegmentation = []
epochs_for_fullSegmentation = [0,1,5,10,99]
size_test_minibatches = 1
saveSegmentation = True

threshold_EARLY_STOP = 0.01

#################################################################################################3

comments = ''


if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)

penalty_MATRIX = np.eye(output_classes,output_classes,dtype='float32')



