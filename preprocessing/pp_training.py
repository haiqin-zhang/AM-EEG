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

""" 
Processes data from training periods. 
Can handle participants recorded on both computers by changing mode to "old" or "new"

"""

#======================================================================================
#                        SPECIFY SUBJECTS 
#======================================================================================
subjects_to_process = ['13']
bad_chs = []
check_events = True
mode = 'new' #old for subs 01 to 12, new for others

eeg_name = 'eeg_training'


#                       PREPROCESSING PARAMETERS
#======================================================================================
plot = False
FS_ORIG = 2048  # Hz

# Printing general info
print_info = False

# Notch filtering
notch_applied = True
freq_notch = 50

# Bandpass filtering
bpf_applied = True
freq_low   = 1
freq_high  = 30
bandpass = str(freq_low) + '-' + str(freq_high)
ftype = 'butter'
order = 3

# Spherical interpolation
int_applied = False
interpolation = 'spline'

# Rereferencing using average of mastoids electrodes
reref_applied = True
reref_type = 'Mastoids'  #Mastoids #Average

# Downsampling
down_applied = True
downfreq = 128
if not down_applied:
    downfreq = FS_ORIG
downfreq_factor = int(FS_ORIG/downfreq)


#                        INITIALIZE DIRECTORIES
#======================================================================================
root_dir = "/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_raw" #where the raw bdf files are
output_base = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_preprocessed_{freq_high}Hz' #where all the preprocessed .mat files and other info go


#====================================================================================== 
#                           LOOP THROUGH SUBJECTS
#======================================================================================

for subject in subjects_to_process:

    file = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/data_raw/sub_{subject}/sub_{subject}_02.bdf'
    print('Processing file: ', file)

    #====================================================================================== 
    #                           check events
    #======================================================================================

    if check_events == True:

        #load file
        raw = mne.io.read_raw_bdf(file, eog=None, misc=None, stim_channel='Status', 
                                    infer_types=False, preload=True, verbose=None)
        #mark bad channels
        raw.info['bads'] = bad_chs

        events = mne.find_events(raw, stim_channel='Status', shortest_event=1)

        if mode == 'old':
            events_2, events_3, events_4, events_5, events_6, trial_starts = sort_events(events)
        elif mode == 'new':
            events_2, events_3, events_4, events_5, events_6, trial_starts, events_MIDI = sort_events_MIDI(events)


        #check events
        plot_subset = False
        start = 0
        end = 100


        plt.figure(figsize = (20,10))

        if plot_subset:
        #keystrokes
            plt.eventplot(events_2[:,0][start:end], lineoffsets = 0)
            plt.eventplot(events_3[:,0][start:end], color = 'green', lineoffsets = -2)
            plt.eventplot(events_4[:,0][start:end], color = 'orange', lineoffsets = -3)
            plt.eventplot(events_5[:,0][start:end], color = 'red', lineoffsets=-4)
        else:
            plt.eventplot(events_2[:,0], lineoffsets = 0)
            plt.eventplot(events_3[:,0], color = 'green', lineoffsets = -2)
            plt.eventplot(events_4[:,0], color = 'orange', lineoffsets = -3)
            plt.eventplot(events_5[:,0], color = 'red', lineoffsets=-4)
        plt.title(f'Events subject {subject} training')
        plt.show()

        df_pre = pd.DataFrame()
        subject_ID = file.split('/')[-2][-2:]
        if subject_ID == '03':
            FS_ORIG = 1024


        output_dir = os.path.join(output_base, subject_ID)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    data = raw.copy()
    

    #======================================================================================
    #                        PREPROCESS
    #======================================================================================
    df_pre = pd.DataFrame()

    output_dir = os.path.join(output_base, subject_ID)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check metadata
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    n_chan = len(ch_names) 


    #======================================================================================
    #                       FILTERING
    #======================================================================================

    ## -------------
    ## Select channels
    ## -------------

    #eeg_channels = ch_names[:66] + [ch_names[-1]]
    eeg_channels = ch_names[0:72]
    eeg = data.copy().pick_channels(eeg_channels)
    if plot:
        eeg.plot(start=100, duration=10, n_channels=len(raw.ch_names))

    ## -------------
    ## Notch filtering
    ## -------------
    df_pre['notch_applied'] = [notch_applied]
    if notch_applied:
        eeg = eeg.notch_filter(freqs=freq_notch)
        df_pre['notch'] = [freq_notch]
        if plot:
            eeg.plot()

    ## -------------
    ## BPFiltering
    ## -------------
    df_pre['bpf_applied'] = [bpf_applied]
    if bpf_applied:
        iir_params = dict(order=order, ftype=ftype)
        filter_params = mne.filter.create_filter(eeg.get_data(), eeg.info['sfreq'], 
                                                l_freq=freq_low, h_freq=freq_high, 
                                                method='iir', iir_params=iir_params)

        if plot:
            flim = (1., eeg.info['sfreq'] / 2.)  # frequencies
            dlim = (-0.001, 0.001)  # delays
            kwargs = dict(flim=flim, dlim=dlim)
            mne.viz.plot_filter(filter_params, eeg.info['sfreq'], compensate=True, **kwargs)
            # plt.savefig(os.path.join(output_dir, 'bpf_ffilt_shape.png'))

        eeg = eeg.filter(l_freq=freq_low, h_freq=freq_high, method='iir', iir_params=iir_params)
        df_pre['bandpass'] = [iir_params]
        df_pre['HPF'] = [freq_low]
        df_pre['LPF'] = [freq_high]
        if plot:
            eeg.plot()

        
        ## -------------
        ## Interpolation
        ## -------------
        df_pre['int_applied'] = [int_applied]
        if int_applied: 
            eeg = eeg.interpolate_bads(reset_bads=False)  #, method=interpolation

            # Get the indices and names of the interpolated channels
            interp_inds = eeg.info['bads']
            interp_names = [eeg.info['ch_names'][i] for i in interp_inds]

            # Print the number and names of the interpolated channels
            print(f'{len(interp_inds)} channels interpolated: {interp_names}')

            df_pre['interpolation'] = [interpolation]
            df_pre['interp_inds'] = [interp_inds]
            df_pre['interp_names'] = [interp_names]

            if plot:
                eeg.plot()
            
        
    ## -------------
    ## Rereferencing
    ## -------------
    df_pre['reref_applied'] = [reref_applied]
    if reref_applied:
        # Set electrodes for rereferencing
        if reref_type == 'Mastoids':
            if 'M1' in eeg.ch_names:
                reref_channels = ['M1', 'M2']
            else: 
                reref_channels = ['EXG1', 'EXG2']
        else:
            reref_channels = 'average'           

        # Actually r-referencing signals
        eeg = eeg.set_eeg_reference(ref_channels=reref_channels)
        df_pre['reref_type'] = [reref_type]
        df_pre['reref_channels'] = [reref_channels]
        if plot:
            eeg.plot()

        
    ## -------------
    ## Downsampling
    ## -------------
    df_pre['down_applied'] = [down_applied]
    df_pre['downfreq'] = [downfreq]
    if down_applied:
        eeg = eeg.resample(sfreq=downfreq)
        if plot:
            eeg.plot()


    #======================================================================================
    #                      UPDATE TRIGGERS
    #======================================================================================

    #create support vectors
    #zero array with downsampled dimensions
    events_original = np.zeros((5, data.get_data().shape[1])) #5 rows = event types, along time axis zero when there is an event and 1 everywhere else


    #get triggers
    section_triggers = mne.find_events(data, stim_channel='Status', shortest_event=1)

    if mode == 'old': 
        events_2, events_3, events_4, events_5, events_6, section_start = sort_events(section_triggers, clean = True)
    elif mode == 'new':
        events_2, events_3, events_4, events_5, events_6, section_start, events_MIDI = sort_events_MIDI(events)


    #get trial start time

        
    try:
        section_start = section_start[0][0]
    except IndexError: 
    # section_start = events_6[0][0]
        section_start = events_2[0][0]


    #find indices corrected for start time
    indices_2 = (events_2[:,0]) 
    indices_3 = (events_3[:,0]) 
    indices_4 = (events_4[:,0]) 
    indices_5 = (events_5[:,0])

    #populate event array with 1s where there are events

    events_original[0][indices_2] = 1
    events_original[1][indices_3] = 1
    events_original[2][indices_4] = 1
    events_original[3][indices_5] = 1

    #resample while preserving events
    cropped_length = eeg.get_data().shape[1]
    events_resampled = np.zeros((5, cropped_length))
    for row in range(events_resampled.shape[0]):
        events_resampled[row] = discretize(events_original[row], final_length = cropped_length, downfreq_factor = downfreq_factor)

    #======================================================================================
    #                       SAVING CROPPED FILES
    #======================================================================================

    eeg_tosave = eeg.get_data()


    savemat(os.path.join(output_dir,f'{eeg_name}_{subject_ID}.mat'), 
            {'all_electrodes': eeg_tosave[0:72], 
                'trial_data': eeg_tosave[0:64, :], 
                'trial_mastoids': eeg_tosave[64:66, :], 
                'trial_exg': eeg_tosave[66:72, :], 
                'events': events_resampled})

        ## -------------
    ## Save preprocessing stages
    ## -------------

    df_pre.to_csv(os.path.join(output_dir, f"preprocess_record_{eeg_name}_{subject_ID}.csv"), index=False)


        