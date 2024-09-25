"""
Takes preprocessed and segmented EEG and eliminates eye blinks by 
automatic ICA component elimination. 

Outputs files to ./data_eog_ica. Use these files to test mTRF models to save time

Do not use for ERP analysis. The ERP epooching script already takes care of ICA

"""

import mne
from mne.preprocessing import ICA

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import pickle

import sys
import os
import glob

sys.path.append('../utils')
from pp_utils import *

#PARAMETERS
#======================================================================================


subjects_to_process = [
                       # '01', '02', '04', '05', '06', '07', '08', '09', '10', 
                      # '11', '12', '13','14','15','16', '17', 
                       #'18','19'
                        '20','21'
                        ]
periods = ['pre', 'post']
#tasks = ['listen', 'motor', 'error']
tasks = ['listen']

overwrite = False

#====================================================================================== 
#                           FILES AND DIRECTORIES
#======================================================================================


root_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed" #dir for files preprocessed and segmented by task
output_base = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_eog_ica' #output folder for ICA


with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)
ch_names_64 = ch_names_all[0:64]
ch_names_72 = ch_names_all[0:72]
#====================================================================================== 
#                           LOOP THROUGH SUBJECTS
#======================================================================================


for subject in subjects_to_process:
    for period in periods: 
        for task in tasks:
            file = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed/{subject}/eeg_{task}_{period}_{subject}.mat'

        print('Processing file: ', file)

        output_dir = os.path.join(output_base, subject)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f'eeg_eogica_{task}_{period}_{subject}.mat')
        if os.path.exists(output_file) and overwrite == False:
            print('Skipping file; file already exists. To overwrite, set overwrite = True.')
            continue

        #LOAD PREPROCESSED DATA
        ##############################################
        data = loadmat(file)
        eeg = data['trial_data']
        events_sv = data['events']
        all_electrodes = data['all_electrodes']
        #
        # eeg = eeg.T #because first dimension has to be time
        fs = 128


        #making raw object from .mat data
        info = mne.create_info(ch_names=ch_names_72, sfreq = 128, ch_types='misc')
        raw = mne.io.RawArray(all_electrodes, info = info)

        #define channel types
        raw.set_channel_types({name: 'eeg' for name in ch_names_72[0:64]})
        raw.set_channel_types({name: 'eog' for name in ch_names_72[66:68]})


        #ICA on EEG data to remove eyeblinks
        #############################################
        ica = ICA(max_iter="auto", random_state=97)
        ica.fit(raw)

        #exclude components
        eog_indices, eog_scores = ica.find_bads_eog(raw)

        ica.exclude = eog_indices
        reconst_raw = raw.copy()
        ica.apply(reconst_raw)

        print('Excluded components', eog_indices)

        #define eeg matrix
        eeg = reconst_raw.get_data()[0:64].T
        #eeg_tosave = eeg.get_data()

        
       
        savemat(output_file, 
                    {'eeg': eeg, 
                     'events': events_sv,
                     'dropped_icas': eog_indices
                    }
                )
        