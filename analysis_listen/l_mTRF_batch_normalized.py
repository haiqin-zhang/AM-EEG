"""
mTRF on normalized eeg data. Expects eeg data to be a list of np arrays with dimensions n_times, n_channels

TO ADD: online normalization of events 
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
from normalize_eeg import *

import pickle
import os



#PARAMETERS AND SUBJECTS
########################################################
subjects_to_process = [
                       '01', '02'
                    ,'04', '05',
                     '06', '07', '08', '09', '10'
                      ,'11', '12', '13','14','15','16', '17', '18'
                        ,'19'
                        , '20', '21'
                       ]


periods = ['pre','post']
task = 'listen'
features = 'onsets' #AM or onsets

overwrite = True
n_segments = 12


#OPENING SUPPORTING FILES
#######################################################

with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)
ch_names_64 = ch_names_all[0:64]
ch_names_72 = ch_names_all[0:72]

fs = 128


for period in periods:
    eeg_list_normalized, stimuli_list_normalized = normalize_eeg_stimuli(subjects_to_process, task, period, features = features)

    for i, subject in enumerate(subjects_to_process):
        #OPENING, PROCESSING EEG DATA
        #######################################################

        eeg = eeg_list_normalized[i]
        stimuli_sv = stimuli_list_normalized[i]

        #directories
        
        
        mTRF_path = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_listen/listen_mTRF_data/listen_mTRF_data_normalized_{features}'
        if not os.path.exists(mTRF_path):
            os.makedirs(mTRF_path)
            print('dir made')

        mTRF_file = f'{mTRF_path}/mTRF_normalized_listen_{period}_{subject}.mat'
        if overwrite == False and os.path.exists(mTRF_file): 
            print('mTRF files already exist. Overwrite not activated.')
            continue


        #################################
        # TO DO: ADD SURPRISAL, MIDI VALUES, AND OTHERS
        # if mTRF factors = MIDI:
            #...
            #stimulus = surprisal_onset_segments etc
        
        #SEGMENTING DATA
        eeg_segments = segment(eeg, n_segments)
        events_segments = segment(stimuli_sv, n_segments)


        #TRAIN TRF
        #################################

        
        tmin, tmax =-0.1, 0.3  # range of time lag
        regularizations= [0.0001, 1, 100, 10000, 1000000]


        #training TRF model
        fwd_trf = TRF(direction=1)
        r_fwd, lambdas = nested_crossval(fwd_trf, events_segments, eeg_segments, fs, tmin, tmax, regularizations)



        #save mTRF
        savemat(mTRF_file, {'weights': fwd_trf.weights, 'r': r_fwd})


