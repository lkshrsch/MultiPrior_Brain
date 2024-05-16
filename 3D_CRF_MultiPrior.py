# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:28:51 2019

@author: hirsch

@author: https://github.com/lucasb-eyer/pydensecrf/tree/94d1cddab277e6e93812dfe7daca7d4693560758

Adapted from the inference.py to demonstate the usage of the util functions.
"""

import pydensecrf.densecrf as dcrf
import numpy as np
import nibabel as nib
from pydensecrf.utils import compute_unary,unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.transform import resize

rawMRI = '/../T1.nii'
segmentation = '/../segmentation.nii'

method = 'All' # Choose either 'All', 'smooth' or 'sharpen' 
grid_search = False


def run_3D_CRF(original_image, segmentation, R_gaussian=2, R_bilat=2, s_chan=3, weight_Potts=3, weight_Potts_gaussian=1, steps=2, model = 'sharpen'):
    
    nii = nib.load(original_image)
    img = nii.get_data()
    img = np.array(img, dtype='float32')    
    nii = nib.load(segmentation)
    labels = nii.get_data()  
    labels += 1
    labels = np.array(labels, dtype='int8')
    M = len(np.unique(labels))
    img = resize(img, labels.shape, preserve_range=True, anti_aliasing=True, order=1)    
    labels_temp = np.unique(labels)
    labels_Dict = {}
    for i in range(0,M):
      labels_Dict[labels_temp[i]] = i+1
    for key in labels_Dict.keys():
      labels[labels == key] = labels_Dict[key]
      
    print('{} Classes found.'.format(len(np.unique(labels))))  
    ###########################
    ### Setup the CRF model ###
    ###########################
    
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[0] * img.shape[1] * img.shape[2], M)
    
    # get unary potentials (neg log probability)
    #U = compute_unary(labels, M)
    U = unary_from_labels(labels, M, gt_prob=0.6)
    d.setUnaryEnergy(U)
         
    Sharpening_Model =  np.array([ [-2., 3., 3.,  3.,  3.,  3,   0.],
                                   [ 3.,-1., 3.,  3.,  3.,  0,   3.],
                                   [ 3., 3.,-1.,  0.,  0.,  1,   3.],
                                   [ 3., 3., 0., -1.,  1.,  1,   3.],
                                   [ 3,  3., 0,   1,  -1,   0.,  3.],
                                   [ 3., 0., 1.,  1.,  0., -1.,  0.],
                                   [ 0., 3., 3.,  3.,  3.,  0., -1.]], dtype='float32')
     

    Smoothing_Model =  np.eye(7, dtype='float32')*-1
       
    if model == 'smooth': 
      my_Model1 = Smoothing_Model
    elif model == 'sharpen': 
      my_Model1 = Sharpening_Model
                 
    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(1,R_gaussian,R_gaussian), shape=img.shape[:])
    d.addPairwiseEnergy(feats, compat=my_Model1*weight_Potts_gaussian,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)    
    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(1,R_bilat,R_bilat), schan=(s_chan), img=img, chdim=-1)
    d.addPairwiseEnergy(feats, compat= my_Model1*weight_Potts,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    ####################################
    ### Do inference and compute map ###
    ####################################
    Q = d.inference(steps)
    
    my_map = np.argmax(Q, axis=0).reshape(img.shape[:])
    if len(np.unique(my_map)) == 1:
      print('Removed everything.. failed.')
      print(np.unique(my_map))
    img_out = np.array(my_map, dtype='int16')

    out = nib.Nifti1Image(img_out, nii.affine)
    out_file = '/'.join(segmentation.split('/')[0:-1]) +'/' + segmentation.split('/')[-1].split('.')[0] + '_CRF.nii'
    nib.save(out, out_file)


#%%
  
if method == 'All':
    run_3D_CRF(rawMRI,segmentation, R_gaussian=2, R_bilat=4, s_chan=1, weight_Potts=4, weight_Potts_gaussian = 1, steps=5, model='sharpen')
    segmentation = segmentation.split('.nii')[0] + '_CRF.nii'
    run_3D_CRF(rawMRI,segmentation, R_gaussian=2, R_bilat=1, s_chan=1, weight_Potts=1, weight_Potts_gaussian = 2, steps=2, model='smooth')#5)
  
if method == 'smooth':
  # Smoothing
    run_3D_CRF(rawMRI,segmentation, R_gaussian=2, R_bilat=1, s_chan=1, weight_Potts=1, weight_Potts_gaussian = 2, steps=2, model='smooth')#5)
  
if method == 'sharpen': 
    # Sharpening
  
    #segmentation = '/home/hirsch/Documents/projects/MSKCC/Segmenter_HumanPerformance/segmentations_MultiPriors_May22/t1post_MSKCC_16-328_1_00533_20020530_t1post-r.nii_r_epoch86.nii.gz'  
    #rawMRI = '/media/hirsch/RNN_training/alignedNii/MSKCC_16-328_1_00533_20020530/sub-r.nii.gz'
    run_3D_CRF(rawMRI, segmentation, R_gaussian=1, R_bilat=4, s_chan=1, weight_Potts=2, weight_Potts_gaussian=0, steps=1, model='sharpen')

    #seg = keep_high_intensity(segmentation, rawMRI)

  
if grid_search:
    # grid search
    for a in [1,2,3,4]:
        for b in [1,2,4,5]:
            for c in [1,2,3,4]:
                #for a2 in [3,4,5]:
                run_3D_CRF(rawMRI,segmentation,a,b,c,a2 = 3)
  