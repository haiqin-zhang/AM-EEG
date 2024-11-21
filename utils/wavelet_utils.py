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

    wavelet_transform: MEAN absolute values of wavelet transform in one channel. size spect_freqs x spect_height
    freqs_to_sum: Range of freqs to sum over from low to high bound.
    freqs_array: the array of frequencies associated with the wavelet transform
    ----
    returns sum: array of size n_channels
    """
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
