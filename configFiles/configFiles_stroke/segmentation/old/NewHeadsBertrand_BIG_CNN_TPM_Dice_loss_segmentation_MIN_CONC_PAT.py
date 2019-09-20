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
    
segmentChannels = ['/CV_folds/stroke/new_heads.txt']
segmentLabels = ''

output_classes = 7
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'
model = 'BIG_CNN_TPM'
dpatch=61

path_to_model = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/training_sessions/BIG_multiscale_CNN_TPM_fullHeadSegmentation_configFile0_BIG_CNN_TPM_Dice_loss_2018-10-02_1846/models/le_CNN_TPM_fullHeadSegmentation_configFile0_BIG_CNN_TPM_Dice_loss_2018-10-02_1846.log_epoch15.h5'

session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
test_subjects = 3
n_fullSegmentations = test_subjects
size_test_minibatches = 500
saveSegmentation = True

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

