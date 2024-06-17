import mne
import os
import glob
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
import numpy as np

# Bandpass filtering
bpf_applied = True
bandpass = '1-8Hz'
freq_low   = 1
freq_high  = 8
ftype = 'butter'
order = 3

root_dir = "/mnt/c/Users/gcantisani/Documents/Datasets/Dataset_song/EEG_recordings/responses/"
files = glob.glob(os.path.join(root_dir, '**', '*.bdf'), recursive=True)
for idx, file in enumerate(files):
    print(idx, file.split('/')[-1])

# Select which file to open - to be changed with a foor loop and create corrisponding output folder
idx_file_to_open = input('Enter idx of subject to open: ')
file_to_open = files[int(idx_file_to_open)]

# Load EEG data
eeg = mne.io.read_raw_bdf(file_to_open, eog=None, misc=None, stim_channel='auto', 
                        infer_types=False, preload=False, verbose=None)
print(eeg)
eeg.load_data()

iir_params = dict(order=order, ftype=ftype)
filter_params = mne.filter.create_filter(eeg.get_data(), eeg.info['sfreq'], 
                                        l_freq=freq_low, h_freq=freq_high, 
                                        method='iir', iir_params=iir_params)

flim = (1., eeg.info['sfreq'] / 2.)  # frequencies
dlim = (-0.001, 0.001)  # delays
kwargs = dict(flim=flim, dlim=dlim)
mne.viz.plot_filter(filter_params, eeg.info['sfreq'], compensate=True, **kwargs)
plt.savefig("../../bpf_ffilt_shape.png") 