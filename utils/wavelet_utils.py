import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import scipy
import pywt
import mne

import sys
from tqdm import tqdm


sys.path.append('../utils')
from ERP_utils import *
from update_sub_lists import *
import glob
import os


def morwav_trans(timeseries, centerfreq, bandwidth, fs = 128, scale_values = None):
    """ 
    timeseries: a n_timepoints x 1 array
    centerfreq: low centerfreq = small gaussian window, less localized in the freq domain
    bandwidth: the freqs of the wavelet (i.e. how localized in the time domain. Large bandwidth = high temporal precision, low freq res)

    ---
    returns:
    cwtmatr: continuous wavelet transform matrix of size n_freqs x n_timepoints. Imaginary values included.
    freqs: vector of freqs converted from sampling_period and scaling factors
    wavelet: the specified wavelet 
    
    """
    wavelet = f"cmor{centerfreq}-{bandwidth}" #use morelet wavelet

    if scale_values == None:
        scales = np.geomspace(6, 130, 40) #scaling factors. Each channel is the wavelet scaled by some constant. The list of constants is log spaced from 6 to 130
    elif isinstance(scale_values, list):
        scales = np.geomspace(scale_values[0], scale_values[1], scale_values[2])
    sampling_period = 1/fs
    cwtmatr, freqs = pywt.cwt(timeseries, scales, wavelet, sampling_period=sampling_period)

    return cwtmatr, freqs, wavelet

def abs_cwtmatr(cwtmatr):
    """ 
    get the abs value of the matrix
    just use np.abs instead because I don't actually care about getting rid of the last row and colum
    """

    cwtmatr_abs = np.abs(cwtmatr[:-1, :-1])
    return cwtmatr_abs


def plot_scaleogram(cwtmatr, freqs, times, vmax = None):

    """ 
    Plots the scalegram of a wavelet transform given the complex matrix
    times: erp times
    vmax: scaling of colorbar
    """
    #cwtmatr_abs = np.abs(cwtmatr[:-1, :-1]) #take off the last value if pcolormesh needs data to be 1 smaller than axes (times, freqs)

    if np.any(np.iscomplex(cwtmatr)):
        cwtmatr_abs = np.abs(cwtmatr)
    else:
        cwtmatr_abs = cwtmatr

    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size as needed
    pcm = ax.pcolormesh(times, freqs, cwtmatr_abs, vmax = vmax, cmap = 'jet')

    #format axes
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (s)')
    
    
    #format y axis so that it's readable in log
    ax.set_yscale('log')

    
    ax.set_yticks([ 2, 5, 10, 20, 30])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    fig.colorbar(pcm, ax=ax)
    

def pad_erp_times(time_vector, num_extend, fs=128):
    """
    Extends a time vector by adding values on both sides, maintaining the step size.

    time_vector (numpy array): Original time vector.
    num_extend (int): Number of values to add on each side.

    Returns:
        numpy array: Extended time vector.
    """
    # Calculate step size
    step_size = 1/fs

    # Extend the time vector on both ends
    start_extend = np.arange(time_vector[0] - step_size * num_extend, time_vector[0], step_size)
    end_extend = np.arange(time_vector[-1] + step_size, time_vector[-1] + step_size * (num_extend + 1), step_size)

    # Combine the arrays
    extended_time_vector = np.concatenate([start_extend, time_vector, end_extend])

    return extended_time_vector



def sum_wavelet_power(wavelet_data, freqs_to_sum, freqs_array):
    """ 
    Sums the power over time. 
    If a range of freqs is included, takes the total over all the freqs

    wavelet_data: MEAN absolute values of wavelet transform in one channel. size spect_freqs x spect_height
    freqs_to_sum: Range of freqs to sum over from low to high bound.
    freqs_array: the array of frequencies associated with the wavelet transform
    ----
    returns sum: array of size n_channels
    """

    ## add assert for shape of wavelet data
    assert len(wavelet_data.shape) == 2, f'wavelet data should be 2-dim, freqs x times. Your data shape: {wavelet_data.shape}'
    freq_idx = time_index_custom(freqs_to_sum, freqs_array)

    wavelet_sub = wavelet_data[np.min(freq_idx): np.max(freq_idx)+1, :] #includes the last index. Can adjust later
    power_sum = np.sum(wavelet_sub)

    return power_sum

def sum_over_channels(wavelet_data, freqs_to_sum, freqs_array):
    """ 
    Loops over listed channels and finds the summed power for each wavelet associated with the channel
    Use for plotting topomaps. Currently only handles summing all 64 EEG channels.

    wavelet_data: 
    """
    ch_idx = np.arange(64) #assumes 64 channels are available

    power_sum_all_ch = np.zeros((64))
    for ch in ch_idx:
        wavelet_data_ch = wavelet_data[ch, :, :]
        power_sum_ch = sum_wavelet_power(wavelet_data_ch, freqs_to_sum, freqs_array)
        power_sum_all_ch[ch] = power_sum_ch
    
    return power_sum_all_ch


def plot_power_freqs(freqs, power_data, label = None, color = None):
    """
    makes a plot of power vs freq
    freqs: x axis data (vector of freqs from wavelet info)
    power_data: 2d array n_subjects x n_freqs
    
    """
    if isinstance (power_data, pd.DataFrame):
    # Assuming power_df_all['power'] is a 2D array where rows are observations (e.g., trials or subjects) and columns are frequencies
        power = np.array(power_data['power'].tolist())  # Convert to numpy array if it's not already
    else: 
        assert isinstance(power_data, np.ndarray)
        power = power_data
    power_mean = power.mean(axis=0)  # Mean across observations
    power_sem = power.std(axis=0, ddof=1) / np.sqrt(power.shape[0])  # SEM computation

    # Plot the mean power spectrum
    plt.plot(freqs, power_mean, label = label, color = color)

    # Add SEM as a shaded region
    plt.fill_between(freqs, 
                    power_mean - power_sem, 
                    power_mean + power_sem, 
                    color = color,
                    alpha=0.3)
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')


def power_over_subs(subjects_to_process, wavelet_dir, freqs_to_sum, already_ave = False, ch_to_process = None):

    """ 
    NOTES: Currently assumes that all data contains only one channel
    Computes power at each frequency for each subject. Takes square of magnitude.
    Averages the wavelets over trials before processing if not already averaged.

    wavelet_dir: dir with one wavelet transform file per subject. Files should contain magnitudes.
    freqs_to_sum: Range of freqs to sum over from low to high bound, or 'all'
        if 'all', the dataframe will return power as arrays of size freqs x power
    already_ave: if data was averaged already
    
    ---
    returns: dataframe with colums for subjects, periods, musicianship, and summed power at each frequency
    dims of summed power: 1 x n_freqs if freqs_to_sum = 'all'. Otherwise a float.
    """

    
    info_path = os.path.join(wavelet_dir, "wavelet_record.mat")
    wavelet_trans_info = loadmat(info_path)
    freqs = wavelet_trans_info['freqs'][0]

    power_df = pd.DataFrame(columns=['subject', 'musician', 'period', 'power'])
    _, _, musicians, _ = load_subject_lists()

    for wavelet_file in sorted(os.listdir(wavelet_dir)):
        sub_id = wavelet_file.split(".")[0].split('_')[-1]
        period =  wavelet_file.split(".")[0].split('_')[-2]

        if sub_id not in subjects_to_process:
            print(f'skipping sub {sub_id}')
            continue
        
        #loading data
        data = loadmat(os.path.join(wavelet_dir, wavelet_file))
        wavelet_sub_mag = data['wavelet_transform']

        #square magnitude data to get power
        wavelet_sub = np.square(wavelet_sub_mag)
        
        #averages data only if it has not already been averaged 
        if not already_ave: 
            wavelet_sub_ave = np.mean(wavelet_sub, axis = 0).squeeze()
        else: 
            wavelet_sub_ave = wavelet_sub
            
        #ensures that only one channel is present in data

        if ch_to_process != None and ch_to_process != 'all': 
            ch_idx = ch_index(ch_to_process)
            wavelet_sub_ave = wavelet_sub_ave[ch_idx, :,:].squeeze()
            
        assert len(wavelet_sub_ave.shape) == 2, f'wavelet data should contain one channel only. Data shape is {wavelet_sub_ave.shape}'
        

        #take subset that is relevant for the frequency
        if isinstance(freqs_to_sum, str) and freqs_to_sum == 'all':
            power_arr = np.zeros(wavelet_sub_ave.shape[0])
            for i, freq in enumerate(freqs): 
                
                power = sum_wavelet_power(wavelet_sub_ave, [freq], freqs)
                power_arr[i] = power
        else:
            power_arr = sum_wavelet_power(wavelet_sub_ave, freqs_to_sum, freqs)
            
        
        #add to dataframe
        power_df_sub = pd.DataFrame({
            'subject': sub_id,
            'period': period,
            'power': [power_arr]

        })
        if sub_id in musicians:
            power_df_sub['musician'] = 1
        else: 
            power_df_sub['musician'] = 0
        power_df = pd.concat([power_df, power_df_sub])

    return power_df
            

def power_over_channels(subjects_to_process, wavelet_dir, freq_band, period = None):
    """ 
    Extracts the power at each channel in a specified band, makes the data ready to topoplot
    freq_band: list of freqs representing lower and upper bound
    period: 'pre' or 'post'. If None, combines the two periods
    ---
    Returns array of size n_subjects x n_channels
    """
    info_path = os.path.join(wavelet_dir, "wavelet_record.mat")
    wavelet_trans_info = loadmat(info_path)
    freqs = wavelet_trans_info['freqs'][0]

    _, _, musicians, _ = load_subject_lists()

    wavelet_sum_list = []
    for wavelet_file in sorted(os.listdir(wavelet_dir)):
            
        sub_id = wavelet_file.split(".")[0].split('_')[-1]
        if sub_id not in subjects_to_process:
            print(f'skipping sub {sub_id}')
            continue
        
        #check the period. 
        if period != None:
            sub_period =  wavelet_file.split(".")[0].split('_')[-2]
            if sub_period != period:
                continue
    
        
        data = loadmat(os.path.join(wavelet_dir, wavelet_file))
        wavelet_sub = data['wavelet_transform']
      

        freq_idx = index_custom(freq_band, freqs)
        wavelet_sub_freq = wavelet_sub[:, np.min(freq_idx):np.max(freq_idx), :]
        
        wavelet_sum = np.sum(wavelet_sub_freq, axis = (1,2))
        wavelet_sum_list.append(wavelet_sum)
    
    wavelet_sum_arr = np.array(wavelet_sum_list)
    
    return wavelet_sum_arr



