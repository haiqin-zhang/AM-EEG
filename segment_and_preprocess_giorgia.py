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

################
## Parameters
################
# General
root_dir = "/mnt/c/Users/gcantisani/Documents/Datasets/Dataset_song/EEG_recordings/responses/"
output_base = '../../outputs/EEG/'
csv_general_path = '../../outputs/features/len_stimuli.csv'
csv_sequence = 'stimuli_sequence.csv'
csv_behave = 'answers_to_questions.csv'
csv_pre = 'preprocessing_pipeline.csv'
plot = False
FS_ORIG = 2048  # Hz
subjects_to_process = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']  

# Preprocessing
# Notch filtering
notch_applied = False
freq_notch = 50

# Bandpass filtering
bpf_applied = True
freq_low   = 0.01
freq_high  = 8
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
downfreq = 64
if not down_applied:
    downfreq = FS_ORIG

# Load stimuli details
df_general = pd.read_csv(csv_general_path) 

############################################################################
## Loop over .bdf recordings
############################################################################
# Find BioSemi files in root_dir and itrate over them
files = glob.glob(os.path.join(root_dir, '**', '*.bdf'), recursive=True)
for idx, file in enumerate(files):
    subject_ID = file.split('/')[-1].split('.')[0]
    if subject_ID not in subjects_to_process:
        continue
    print(idx, file.split('/')[-1])

    # Select which file to open
    file_to_open = file

    # Create output folder
    name_subject = file_to_open.split('/')[-1].split('.')[0]
    output_dir = os.path.join(output_base, bandpass, reref_type, name_subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create file that keeps track of the preprocessing
    df_pre = pd.DataFrame()

    # Load and resave stimuli sequence
    path = file_to_open.split('/')
    path[-1] = csv_sequence
    path = '/'.join(path)
    df_stimuli = pd.read_csv(path)  
    df_stimuli.to_csv(os.path.join(output_dir, csv_sequence), index=False)

    # Load and resave behavioural questions
    path = file_to_open.split('/')
    path[-1] = csv_behave
    path = '/'.join(path)
    if os.path.exists(path):
        df_behave = pd.read_csv(path)  
        df_behave.to_csv(os.path.join(output_dir, csv_behave), index=False)

    ############################################################################
    ## Load EEG data
    ############################################################################
    # Load raw
    raw = mne.io.read_raw_bdf(file_to_open, eog=None, misc=None, stim_channel='auto', 
                              infer_types=False, preload=False, verbose=None)
    print(raw)
    raw.load_data()

    # Check metadata
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    n_chan = len(ch_names) 
    print('the (cropped) sample data object has {} time samples and {} channels.'
        ''.format(n_time_samps, n_chan))
    print('The last time sample is at {} seconds.'.format(time_secs[-1]))
    print('The first few channel names are {}.'.format(', '.join(ch_names[:3])))
    print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
    print(raw.info['sfreq'], 'Hz')            # sampling frequency
    print(raw.info['description'], '\n')      # miscellaneous acquisition info
    print(raw.info)
    if plot:
        raw.plot(start=100, duration=10)

    ############################################################################
    ## Find events in trigger channel
    ############################################################################
    # Find starting sample of events in the trigger channel
    N_start_events = mne.find_events(raw)

    # Select only relevant events
    N_start_events = N_start_events[N_start_events[:, 2] == 65281]

    # Get starting samples of events
    N_start_events = N_start_events[:, 0]

    # Little exception for subject 16
    if subject_ID == '16':
        # Select only the triggers of session 2 - discrad first 6
        N_start_events = N_start_events[6:]       

    # Get corresponding time
    T_start_events = N_start_events / raw.info['sfreq']

    # Check there are as many triggers as the listened stimuli
    assert len(N_start_events) == len(df_stimuli)

    ############################################################################
    ## Segment and then process EEG data
    ############################################################################
    for index, row_stimuli in df_stimuli.iterrows():

        ###############################
        ## Crop data
        ###############################
        # Get the name of the listened stimulus
        filename = row_stimuli['File'].split('.')[0]
        
        # Get corresponding metadata from the generl dataframe
        row_stimulus = df_general.loc[df_general['Filename'] == filename]
        
        # Get the onset and duration in seconds and in samples
        onset_seconds = T_start_events[index] 
        duration_seconds = row_stimulus['T'].values[0]
        eeg = raw.copy().crop(tmin=onset_seconds, tmax=onset_seconds+duration_seconds)

        ###############################
        ## Preprocessing
        ###############################
        ## -------------
        ## Select channels
        ## -------------
        eeg_channels = ch_names[0:66]
        eeg = eeg.pick_channels(eeg_channels)
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
        ## Intrpolation
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
                # Little exception for subject 1 as channel names are messed up
                if name_subject == '01':
                    reref_channels = ['M1-0', 'M2-0'] 
                else:
                    reref_channels = ['M1', 'M2']   
            else:
                reref_channels = 'average'           

            # Actually r-referencing signals
            eeg = eeg.set_eeg_reference(ref_channels=reref_channels)
            df_pre['reref_type'] = [reref_type]
            df_pre['reref_channels'] = [reref_channels]
            if plot:
                eeg.plot()

        ## -------------
        ## Resampling
        ## -------------
        df_pre['down_applied'] = [down_applied]
        df_pre['downfreq'] = [downfreq]
        if down_applied:
            eeg = eeg.resample(sfreq=downfreq)
            print(eeg.info)
            if plot:
                eeg.plot()

        ## -------------
        ## Save preprocessing stages
        ## -------------
        df_pre.to_csv(os.path.join(output_dir, csv_pre), index=False)


        # Get only data matrix
        eeg = eeg.get_data()
        
        # Save file to mupy in the subject's folder
        print('Saving EEG responses to ', filename, eeg.shape)
        savemat(os.path.join(output_dir, filename + '.mat'), {'trial_data': eeg[0:64, :], 
                                                              'trial_mastoids': eeg[64:, :]})