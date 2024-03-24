################
## Imports
################
import mne
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt


#======================================================================================
#                        INITIALIZE DIRECTORIES
#======================================================================================
root_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_raw" #where the raw bdf files are
output_base = '/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed' #where all the preprocessed .mat files and other info go

plot = False
FS_ORIG = 2048  # Hz
#subjects_to_process = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                     #  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']  

subjects_to_process = ['01']
