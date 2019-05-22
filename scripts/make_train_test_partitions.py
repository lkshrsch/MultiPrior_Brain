# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:51:25 2018

@author: hirsch
"""

labels = open('/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/CV_folds/stroke/ALL_LABELS.txt')
labels = labels.readlines()
mris = open('/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/CV_folds/stroke/ALL_MRI.txt')
mris = mris.readlines()

directory = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/CV_folds/stroke/'

indexes =range(len(mris))

from random import shuffle
shuffle(indexes)
len(indexes)
import numpy as np
parts = np.array_split(indexes, 4)
len(parts[0])

n_validation = 4


for x in range(len(parts)):
  test = [mris[i] for i in parts[x]]
  train =  [mris[i] for i in indexes if i not in parts[x]]
  validation = train[-n_validation:]  
  train = train[:-n_validation]
  
  f = open(directory + 'MRIs_test_set{}.txt'.format(x),'a')
  for item in test:
      f.write(item)
  f.close()
  f = open(directory + 'MRIs_train_set{}.txt'.format(x),'a')
  for item in train:
      f.write(item)
  f.close()
  f = open(directory + 'MRIs_validation_set{}.txt'.format(x),'a')
  for item in validation:
      f.write(item)
  f.close()

for x in range(len(parts)):
  labels_test = [labels[i] for i in parts[x]]
  labels_train =  [labels[i] for i in indexes if i not in parts[x]]
  labels_validation = labels_train[-n_validation:]  
  labels_train = labels_train[:-n_validation]  
  
  f = open(directory + 'labels_test_set{}.txt'.format(x),'a')
  for item in labels_test:
      f.write(item)
  f.close()
  f = open(directory + 'labels_train_set{}.txt'.format(x),'a')
  for item in labels_train:
      print(item)
      f.write(item)
  f.close()
  f = open(directory + 'labels_validation_set{}.txt'.format(x),'a')
  for item in labels_validation:
      print(item)
      f.write(item)
  f.close()
    
