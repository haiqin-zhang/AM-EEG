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
subjects_to_process = ['05', '06', '07']
periods = ['pre', 'post']

overwrite = True #overwrite existing files
plot = False

#-----------------------------------------

#get channel names
with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)

ch_names_72 = ch_names_all[0:72]
downfreq = 128


#======================================================================================
#                        INITIALIZE DIRECTORIES
#======================================================================================
pp_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed" #where the preprocessed files are
evokeds_folder = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_error/error_ERP_data'
epochs_folder = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_error/error_epochs_data'


#======================================================================================
#                        CALCULATE EVOKEDS
#======================================================================================
for folder in sorted(os.listdir(pp_dir)):
    if folder not in subjects_to_process:
        continue

    print('\nPROCESSING SUBJECT ', folder)
    sub_pp_dir = os.path.join(pp_dir, folder)
    for period in periods:
        data_path = glob.glob(os.path.join(sub_pp_dir, f'eeg_error_{period}_??.mat'))[0]

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
        t_inv = clean_triggers(events_arr[events_arr[:, 2]==3])
        t_shinv = clean_triggers(events_arr[events_arr[:, 2]==4])
        t_norm = clean_triggers(events_arr[events_arr[:, 2]==5])
        t_modeswitch = np.concatenate([t_inv, t_shinv, t_norm])
        t_modeswitch = events_inorder(t_modeswitch)
        
        #--------------------------------------------
        #               SET UP EVOKEDS OBJECTS
        #--------------------------------------------
        epochs = mne.Epochs(reconst_raw, t_keystrokes, tmin=-0.2, tmax=0.5, preload=True)
        evoked = epochs.average()

        #keystrokes in the inverted mapping
        inv_sections = find_sections(reconst_raw, t_inv, t_modeswitch)
        inv_keystrokes = find_keystrokes(reconst_raw, t_keystrokes, inv_sections)
        epochs_inv = mne.Epochs(reconst_raw, inv_keystrokes, tmin=-0.3, tmax=0.7, preload=True)
        inv_evoked = epochs_inv.average()
        if plot:
            fig = inv_evoked.plot(titles = f'Keystrokes - inverted mapping {period} sub{subject_ID}')

        #keystrokes in shinv mapping
        shinv_sections = find_sections(reconst_raw, t_shinv, t_modeswitch)
        shinv_keystrokes = find_keystrokes(reconst_raw, t_keystrokes, shinv_sections)
        epochs_shinv = mne.Epochs(reconst_raw, shinv_keystrokes, tmin=-0.3, tmax=0.7, preload=True)
        shinv_evoked = epochs_shinv.average()
        if plot:
            fig = shinv_evoked.plot(titles = f'Shifted keystrokes {period} sub{subject_ID}')

        #keystrokes in normal mapping
        norm_sections = find_sections(reconst_raw, t_norm, t_modeswitch)
        norm_keystrokes = find_keystrokes(reconst_raw, t_keystrokes, norm_sections)
        epochs_norm = mne.Epochs(reconst_raw, norm_keystrokes, tmin=-0.3, tmax=0.7, preload=True)
        norm_evoked = epochs_norm.average()
        if plot:
            fig = norm_evoked.plot(titles = f'Keystrokes - normal mapping {period} sub{subject_ID}')


        #--------------------------------------------
        #         SAVE DATA
        #--------------------------------------------
        #ERPs
        mne.write_evokeds(f'{evokeds_folder}/error_ERP_all_{period}_{subject_ID}.fif', evoked, overwrite = overwrite)
        mne.write_evokeds(f'{evokeds_folder}/error_ERP_inv_{period}_{subject_ID}.fif', inv_evoked, overwrite = overwrite)
        mne.write_evokeds(f'{evokeds_folder}/error_ERP_shinv_{period}_{subject_ID}.fif', shinv_evoked, overwrite = overwrite)
        mne.write_evokeds(f'{evokeds_folder}/error_ERP_norm_{period}_{subject_ID}.fif', norm_evoked, overwrite = overwrite)

        #epochs
        epochs.save(f'{epochs_folder}/error_epochs_all_{period}_{subject_ID}.fif', overwrite = overwrite)
        epochs_inv.save(f'{epochs_folder}/error_epochs_inv_{period}_{subject_ID}.fif', overwrite = overwrite)
        epochs_shinv.save(f'{epochs_folder}/error_epochs_shinv_{period}_{subject_ID}.fif', overwrite = overwrite)
        epochs_norm.save(f'{epochs_folder}/error_epochs_norm_{period}_{subject_ID}.fif', overwrite = overwrite)


        

        
