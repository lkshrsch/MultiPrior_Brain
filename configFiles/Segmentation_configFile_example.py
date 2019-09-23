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
 
    
TPM_channel = ' '
    
segmentChannels =  ['/CV_folds/Test_Data.txt']
segmentLabels = '/CV_folds/Test_Labels.txt'

output_classes = 7


############################# MODEL PARAMETERS ###########################
model = 'Baseline'
segmentation_dpatch = 51*3

path_to_model = 'Path/To/Trained/Model.h5'

############################ TEST PARAMETERS ##############################
quick_segmentation = True
softmax_output = True
test_subjects = 15
list_subjects_fullSegmentation = []
size_test_minibatches = 1
saveSegmentation = True

session =  path_to_model.split('/')[-3]

penalty_MATRIX = np.eye(output_classes,output_classes,dtype='float32')

comments = ''

