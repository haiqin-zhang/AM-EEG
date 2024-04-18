import mne
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import pickle

from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne_icalabel import label_components

import sys
sys.path.append('../utils')
from pp_utils import *
from plot_utils import *

#======================================================================================
#                        PARAMETERS
#======================================================================================

#CHANGE THIS AS THE EXPERIMENT PROGRESSES
#----------------------------------------
subjects_to_process = ['08']
periods = ['pre', 'post']
overwrite = True

#----------------------------------------
with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)

ch_names_72 = ch_names_all[0:72]
downfreq = 128
plot = False



#======================================================================================
#                        INITIALIZE DIRECTORIES
#======================================================================================
pp_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed" #where the preprocessed files are
evokeds_folder = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_listen/listen_ERP_data'
epochs_folder = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_listen/listen_epochs_data'


#======================================================================================
#                        CALCULATE EVOKEDS
#======================================================================================
for folder in sorted(os.listdir(pp_dir)):
    
    if folder not in subjects_to_process:
        continue
    
    print('\nPROCESSING SUBJECT ', folder)
    sub_pp_dir = os.path.join(pp_dir, folder)

    for period in periods: 
        data_path = glob.glob(os.path.join(sub_pp_dir, f'eeg_listen_{period}_??.mat'))[0]

        subject_ID = data_path.split('.')[0][-2:]

        print('Opening', data_path.split('/')[-1])
        

        #--------------------------------------------
        #               LOAD FILE AND EVENTS
        #--------------------------------------------
        data = loadmat(data_path)
        eeg = data['trial_data']
        refs = data['trial_mastoids']
        all_electrodes = data['all_electrodes']
        events_sv = data['events']

        #making raw object from .mat data
        info = mne.create_info(ch_names=ch_names_72, sfreq = 128, ch_types='misc')
        raw = mne.io.RawArray(all_electrodes, info = info)

        #define channel types
        raw.set_channel_types({name: 'eeg' for name in ch_names_72[0:64]})
        raw.set_channel_types({name: 'eog' for name in ch_names_72[66:68]})

        raw.set_montage('biosemi64')

        #check psd
        if plot:
            mne.viz.plot_raw_psd(raw, fmin = 0, fmax = 64)  

        #--------------------------------------------
        #               ICA
        #--------------------------------------------
        ica = ICA(max_iter="auto", random_state=97)
        ica.fit(raw)

        #exclude components
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        print('Rejecting components:', eog_indices)
        ica.exclude = eog_indices
        reconst_raw = raw.copy()
        ica.apply(reconst_raw)

        #--------------------------------------------
        #               SET UP TRIGGERS
        #--------------------------------------------
        events_arr = make_raw_events(events_sv)
        t_keystrokes = clean_triggers(events_arr[events_arr[:, 2]==2])

        
        #--------------------------------------------
        #               SET UP EVOKEDS OBJECTS
        #--------------------------------------------
        epochs = mne.Epochs(reconst_raw, t_keystrokes, tmin=-0.2, tmax=0.5, preload=True)
        evoked = epochs.average()

        #plot evoked
        if plot:
            evoked.plot(titles = f'Listening-only keystrokes {period} sub{subject_ID}')
            evoked.plot_topomap(times=[0.1, 0.2, 0.5], average=0.0001)



        #--------------------------------------------
        #         SAVE DATA
        #--------------------------------------------
            
        #better to save as Epochs instead of Evoked if you want to plot CI and do other stuff
        mne.write_evokeds(f'{evokeds_folder}/listen_ERP_{period}_{subject_ID}.fif', evoked, overwrite = overwrite)
        epochs.save(f'{epochs_folder}/listen_epochs_{period}_{subject_ID}.fif', overwrite = overwrite)

        
