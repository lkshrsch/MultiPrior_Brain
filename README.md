# DeepPriors_package
Repository for "Tissue segmentation with deep 3D networks and spatial priors" https://arxiv.org/abs/1905.10010


# Quick Tutorial

Overview of files:

# Train a model with data:

run_Models_stroke.sh  -->  TRAIN_TEST.py 

TRAIN_TEST.py --train_configFile

train_configFile:
- which model to train
- which data to use
- hyperparameter of the training session 


run_SEGMENTATION.sh   --> SEGMENT.py

SEGMENT.py --segment_configFile

segment_configFile:
- Which (trained) model to use
- Which data to segment

############ Directory structure ##############

/configFiles       --> Contains all configuration files  for training and segmenting
/CV_folds          --> Contain the links to all data called through the configuration files
/scripts           --> Contain library of functions and model definitions
/training_sessions --> a folder for each initiated training session called with TRAIN_TEST.py (run_Models_stroke.sh)





To start using in a new environment, one must attach the correct paths to the data in CV_folds. 

ConfigFiles and all other scripts use paths relative to main scripts (TRAIN_TEST.py, SEGMENT.py, MAKE_MODEL.py). So one needs to 
preserve the original package architecture.

Only Data can be placed anywhere else, that is why this has to be linked on the CV_folds lists. Instructions on how to do this are
found in CV_folds/stroke/README


Need also to get correct version of bash script 'run_myModel.sh'

append contents of configFiles into bash script 

just ls with full path -d 1 $PWD *.*
prefix is python TRAIN_TEST.py  (sed -i 's|^|python TRAIN_TEST.py |g' run..sh) 
suffix is ; (sed -i 's|$|;|g' run..sh) 


CHECK that parameters are correct for Azure.

iterations = 5
patches = 50000

do not save segmentations
