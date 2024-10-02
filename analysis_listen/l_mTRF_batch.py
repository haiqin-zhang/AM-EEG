"""
Works on the preprocessed and cropped NON-NORMALIZED data (the normalized data has a different structure so won't work here)

"""


import mne
from mne.preprocessing import ICA


import mtrf
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval
import eelbrain as eel

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile, loadmat, savemat
from scipy.signal import hilbert

import librosa
import librosa.display
import mido

import sys
sys.path.append('../utils')
from pp_utils import *
from plot_utils import *
from mTRF_utils import *
from midi_utils import *

import pickle
import os



#PARAMETERS AND SUBJECTS
########################################################
subjects_to_process = [
                        #'01', '02', '04', '05', '06', '07', '08', '09', '10', 
                    #  '11', '12', '13','14','15','16', '17', '18'
                      #  '19'
                        #'20', '21'
    
    '02', '06', '07', '12', '13', '14', '15', '17', '18', '21'
                       ]

periods = ['pre', 'post']
features = 'AM' #AM or onsets

overwrite = False
n_segments = 12


#OPENING SUPPORTING FILES
#######################################################

with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)
ch_names_64 = ch_names_all[0:64]
ch_names_72 = ch_names_all[0:72]

fs = 128


for subject in subjects_to_process:
    for period in periods:
        #OPENING, PROCESSING EEG DATA
        #######################################################
        #directories
        eeg_path = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_eog_ica/{subject}/eeg_eogica_listen_{period}_{subject}.mat'
        mTRF_path = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_listen/listen_mTRF_data/listen_mTRF_data_{features}'

        mTRF_file = f'{mTRF_path}/listen_{period}_{subject}.mat'
        if overwrite == False and os.path.exists(mTRF_file): 
            print('mTRF files already exist. Overwrite not activated.')
            continue

        #LOAD PREPROCESSED DATA
        data = loadmat(eeg_path)
        eeg = data['eeg']
        events_sv = data['events']


        #OPENING EVENTS
        #################################
        #ONSET
        events_arr = make_raw_events(events_sv)
        events_keystrokes = clean_triggers(events_arr[events_arr[:, 2]==2])
        onset_indices = events_keystrokes[:,0]

        onsets_sv = np.zeros_like(events_sv[0])
        onsets_sv[onset_indices] = 1

        #SURPRISAL
        #idyompy for now but can try DREX
        idyompy_surprisal = loadmat('idyompy_reps.mat')
        surprisal_idyompy_rep4 = idyompy_surprisal['rep4'][0]

        surprise_type = surprisal_idyompy_rep4

        #make support vector for surprisal
        index_surprisal = 0
        surprisal_sv  = []
        for num in onsets_sv:
            if num == 1:
                surprisal_sv.append(surprise_type[index_surprisal])
                index_surprisal += 1
            else:
                surprisal_sv.append(num)
        surprisal_sv = np.array(surprisal_sv)

        #################################
        # TO DO: ADD SURPRISAL, MIDI VALUES, AND OTHERS
        # if mTRF factors = MIDI:
            #...
            #stimulus = surprisal_onset_segments etc
        
        #SEGMENTING DATA
        eeg_segments = segment(eeg, n_segments)
        onset_segments = segment(onsets_sv, n_segments) 
        surprisal_segments = segment(surprisal_sv, n_segments)


        #stacking stimulus segments
        AM_sv = np.vstack([onsets_sv, surprisal_sv]).T
        AM_segments = segment(AM_sv, n_segments)


        #TRAIN TRF
        #################################

        
        tmin, tmax =-0.1, 0.3  # range of time lag
        regularizations= [0.0001, 0.01, 1, 100, 10000, 1000000]

        if features == 'AM': 
            stimulus = AM_segments
        elif features == 'onsets':
            stimulus = onset_segments

        #training TRF model
        fwd_trf = TRF(direction=1)
        r_fwd, lambdas = nested_crossval(fwd_trf, stimulus, eeg_segments, fs, tmin, tmax, regularizations)



        #save mTRF
        savemat(f'{mTRF_path}/listen_{period}_{subject}.mat', {'weights': fwd_trf.weights, 'r': r_fwd})

        """
        This only saves the mTRF but consider saving the entire fwd_trf
        """


