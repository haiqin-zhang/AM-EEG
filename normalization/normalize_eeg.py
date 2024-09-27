"""
Concatenates and normalizes EEG data

Saves a .mat file with task- and period-specific EEG for all participants.
Also optionally saves a .mat file with the nonnormalized, task and period specific EEG concatenated for all participants

"""
import mne
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

import os
import sys
sys.path.append('../utils')
from mTRF_utils import *


#======================================================================================
#                        PARAMETERS
#======================================================================================
subjects_to_process =['01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
task = 'listen'
period = 'post'
overwrite = True
save_concatenated_original = False

#======================================================================================
#                        INITIALIZE DIRECTORIES
#======================================================================================
pp_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed" #where the preprocessed files are
normalized_dir = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/normalization'

#======================================================================================
#                        CONCATENATE AND NORMALIZE
#======================================================================================
#initialize lists for saving
eeg_list = []
eeg_list_normalized = []

for folder in sorted(os.listdir(pp_dir)):
    
    if folder not in subjects_to_process:
        continue
    
    sub_pp_dir = os.path.join(pp_dir, folder)
    print('\nPROCESSING SUBJECT ', folder)


    data_path = glob.glob(os.path.join(sub_pp_dir, f'eeg_{task}_{period}_??.mat'))[0]
    subject_ID = data_path.split('.')[0][-2:]
    print('Opening', data_path.split('/')[-1])
    
    #--------------------------------------------
    #               LOAD FILE AND EVENTS
    #--------------------------------------------
    data = loadmat(data_path)
    eeg = data['trial_data'].T #only the 64 eeg channels, no ext channels

    #append to master list
    eeg_list.append(eeg)



eeg_list_normalized = normalize_responses(eeg_list)


#SAVE FILES
#======================================================================================

file_tosave = os.path.join(normalized_dir,f'normalized_concat_{task}_{period}.mat')
os.path.exists(file_tosave)

mat_tosave = {'eeg_normalized': eeg_list_normalized,
             'subjects': subjects_to_process,
            }
if save_concatenated_original:
    mat_tosave['eeg_original': eeg_list]

if overwrite == False and os.path.exists(file_tosave):
    print('File already exists. Choose another file name or set Overwrite to True')
else:
    savemat(file_tosave, mat_tosave)

            