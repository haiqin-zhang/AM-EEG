
"""
Provides a single function to manually preprocess raw files.
"""

import mne
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append('../utils')
from pp_utils import *



def preprocess_manual(raw, print_info = False, 

    # Notch filtering
    notch_applied = True, 
    freq_notch = 50, 

    # Bandpass filtering
    bpf_applied = True, 
    freq_low   = 0.01,
    freq_high  = 30,
    ftype = 'butter',
    order = 3,

    # Spherical interpolation
    int_applied = False,
    interpolation = 'spline',

    # Rereferencing using average of mastoids electrodes
    reref_applied = True,
    reref_type = 'Mastoids', #Mastoids #Average

    # Downsampling
    down_applied = True,
    downfreq = 128):
    

    raw = mne.read_raw_bdf(raw)

    FS_ORIG = raw.info['sfreq']

    if not down_applied:
        downfreq = FS_ORIG
    downfreq_factor =int(FS_ORIG/downfreq)

    bandpass = str(freq_low) + '-' + str(freq_high)

    