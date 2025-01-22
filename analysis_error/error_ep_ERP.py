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
subjects_to_process =  ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
#subjects_to_process = ['17']
periods = ['pre', 'post']
keystroke_trigs = 'audio' #audio for everyone because we are looking for reaction to audio

overwrite = True #overwrite existing files
plot = False

find_mapchanges = True
find_modekeystrokes = True

#-----------------------------------------

#get channel names
with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)

ch_names_72 = ch_names_all[0:72]
downfreq = 128

#times for cropping ERPs
erp_begin = -0.5
erp_end = 0.5


#======================================================================================
#                        INITIALIZE DIRECTORIES
#======================================================================================
pp_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed_30Hz" #where the preprocessed files are
evokeds_folder = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_error/error_ERP_data_n05to05_30Hz_corrected'
epochs_folder = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_error/error_epochs_data_n05to05_30Hz_corrected'

for folder in [evokeds_folder, epochs_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
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

        if keystroke_trigs == 'MIDI':
            t_keystrokes = clean_triggers(events_arr[events_arr[:, 2]==6])
        else:
            t_keystrokes = clean_triggers(events_arr[events_arr[:, 2]==2])


        t_inv = clean_triggers(events_arr[events_arr[:, 2]==3])
        t_shinv = clean_triggers(events_arr[events_arr[:, 2]==4])
        t_norm = clean_triggers(events_arr[events_arr[:, 2]==5])
        t_modeswitch = np.concatenate([t_inv, t_shinv, t_norm])
        t_modeswitch = events_inorder(t_modeswitch)


        #--------------------------------------------
        #               SET UP EVOKEDS OBJECTS
        #--------------------------------------------

        #all epochs
        epochs = mne.Epochs(reconst_raw, t_keystrokes, tmin=erp_begin, tmax=erp_end, preload=True, reject = None)
        print(epochs.drop_log)
        epochs = epochs.copy().interpolate_bads(reset_bads = True)
        evoked = epochs.average()


        #epochs separated by mode
        if find_modekeystrokes:
            epochs_inv, evoked_inv = epochs_bymode(reconst_raw, t_keystrokes, t_inv, t_modeswitch, erp_begin, erp_end)
            epochs_shinv, evoked_shinv = epochs_bymode(reconst_raw, t_keystrokes, t_shinv, t_modeswitch, erp_begin, erp_end)
            epochs_norm, evoked_norm = epochs_bymode(reconst_raw, t_keystrokes, t_norm, t_modeswitch, erp_begin, erp_end)

                    #ERPs
            mne.write_evokeds(f'{evokeds_folder}/error_ERP_all_{period}_{subject_ID}.fif', evoked, overwrite = overwrite)
            mne.write_evokeds(f'{evokeds_folder}/error_ERP_inv_{period}_{subject_ID}.fif', evoked_inv, overwrite = overwrite)
            mne.write_evokeds(f'{evokeds_folder}/error_ERP_shinv_{period}_{subject_ID}.fif', evoked_shinv, overwrite = overwrite)
            mne.write_evokeds(f'{evokeds_folder}/error_ERP_norm_{period}_{subject_ID}.fif', evoked_norm, overwrite = overwrite)

            #epochs
            epochs.save(f'{epochs_folder}/error_epochs_all_{period}_{subject_ID}.fif', overwrite = overwrite)
            epochs_inv.save(f'{epochs_folder}/error_epochs_inv_{period}_{subject_ID}.fif', overwrite = overwrite)
            epochs_shinv.save(f'{epochs_folder}/error_epochs_shinv_{period}_{subject_ID}.fif', overwrite = overwrite)
            epochs_norm.save(f'{epochs_folder}/error_epochs_norm_{period}_{subject_ID}.fif', overwrite = overwrite)


        if find_mapchanges:
            #keystrokes for keystrokes after map change and other keystrokes
            first_keystrokes = mapchange_keystrokes(t_modeswitch = t_modeswitch, t_keystroke=t_keystrokes)
            other_keystrokes = withinmap_keystrokes(t_keystrokes, first_keystrokes)
   
            #epochs separated by whether it is immediately after a map change or not
            epochs_firsts, evoked_firsts  = construct_ep_ev(reconst_raw, first_keystrokes, erp_begin, erp_end)
            epochs_others, evoked_others = construct_ep_ev(reconst_raw, other_keystrokes, erp_begin, erp_end)

            mne.write_evokeds(f'{evokeds_folder}/error_ERP_firsts_{period}_{subject_ID}.fif', evoked_firsts, overwrite = overwrite)
            mne.write_evokeds(f'{evokeds_folder}/error_ERP_others_{period}_{subject_ID}.fif', evoked_others, overwrite = overwrite)
            epochs_firsts.save(f'{epochs_folder}/error_epochs_firsts_{period}_{subject_ID}.fif', overwrite = overwrite)
            epochs_others.save(f'{epochs_folder}/error_epochs_others_{period}_{subject_ID}.fif', overwrite = overwrite)


        if plot:
            fig = evoked_inv.plot(titles = f'Keystrokes - inverted mapping {period} sub{subject_ID}')
            fig = evoked_shinv.plot(titles = f'Shifted keystrokes {period} sub{subject_ID}')
            fig = evoked_norm.plot(titles = f'Keystrokes - normal mapping {period} sub{subject_ID}')



        

        
