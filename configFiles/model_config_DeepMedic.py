#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
workingDir = os.getcwd()
  
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'
model = 'DeepMedic'
dpatch=51
L2 = 0.0001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice2'

my_custom_objects = {'dice_coef_multilabel2':0,
                     'dice_coef_multilabel0':0,
                     'dice_coef_multilabel1':0}

num_channels = 1
output_classes = 6

dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 2e-04
optimizer_decay = 0

