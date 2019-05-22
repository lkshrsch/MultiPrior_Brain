#!/usr/bin/env bash

#python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile_ALL_BIG_CNN_TPM_Dice_loss.py;
#python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile0_BIG_CNN_TPM_Dice_loss.py;
#python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile0_DeepMedic.py;


python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile0_BIG_singleScale_CNN_TPM_flexibleInput_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile0_BIG_CNN_TPM_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile0_DeepMedic.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile0_BIG_CNN_DUMMY-TPM_Dice_loss.py

python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile1_BIG_singleScale_CNN_TPM_flexibleInput_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile1_BIG_CNN_TPM_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile1_DeepMedic.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile1_BIG_CNN_DUMMY-TPM_Dice_loss.py

python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile2_BIG_singleScale_CNN_TPM_flexibleInput_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile2_BIG_CNN_TPM_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile2_DeepMedic.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile2_BIG_CNN_DUMMY-TPM_Dice_loss.py

python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile3_BIG_singleScale_CNN_TPM_flexibleInput_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile3_BIG_CNN_TPM_Dice_loss.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile3_DeepMedic.py
python TRAIN_TEST.py ./configFiles/configFiles_stroke/configFile3_BIG_CNN_DUMMY-TPM_Dice_loss.py

