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
    
TPM_channel = ''#'/logTPM.nii'
    
segmentChannels =  ['/CV_folds/stroke/MRIs_test_set3.txt']
segmentLabels = '/CV_folds/stroke/labels_test_set3.txt'

output_classes = 7
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic', 'BIG_multiscale_CNN_TPM_flexible', 'BIG_singleScale_CNN_TPM'
model = 'Vanilla'
#dpatch=61
segmentation_dpatch = 57*3

path_to_model = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/training_sessions/Vanilla_fullHeadSegmentation_configFile3_Vanilla_2019-02-20_1651/models/HeadSegmentation_configFile3_Vanilla_2019-02-20_1651.log_epoch11.h5'

session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
test_subjects = 15
n_fullSegmentations = test_subjects
size_test_minibatches = 1
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

