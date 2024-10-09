import numpy as np
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind, kstest, wilcoxon
import pickle
import mne
import pandas as pd

import glob
import os
from update_sub_lists import*

def load_erp_times():
    with open('../utils/erp_times.pkl', 'rb') as file:
        times = pickle.load(file)
        return times

def load_channels():
    #get channel names
    with open('../utils/ch_names.pkl', 'rb') as file:
        ch_names_all = pickle.load(file)

    ch_names_64 = ch_names_all[0:64]
    ch_names_72 = ch_names_all[0:72]
    

    return ch_names_64, ch_names_72

"""
find index of a channel in the data array given the channel name. Relies on the preloaded list of channels.

Takes a list of channel names, e.g. ['Cz']
Returns a list of indices
"""

def ch_index(ch_list): 
    """
    finds indices of channels given the channel names
    ch_list: list of channel names, e.g. ['FCz', 'Cz']
    ---
    returns: list of channel indices
    """
    #get channel names which will be used for indexing channels
    assert isinstance (ch_list, list)
    with open('../utils/ch_names.pkl', 'rb') as file:
        ch_names_all = pickle.load(file)

    ch_names_72 = ch_names_all[0:72]
    ch_indices = [ch_names_72.index(item) if item in ch_names_72 else None for item in ch_list]
    return ch_indices


def time_index(timepoints):

    """
    Finds the index in the data (which should be an array) given a list of timepoints expressed in seconds
    ---
    Returns a list of indices  
    """
    assert isinstance(timepoints, list)
    with open('erp_times.pkl', 'rb') as file:
        erp_times = pickle.load(file)
    idx_list = []
    for time in timepoints: 
        time_idx = min(range(len(erp_times)), key=lambda i: abs(erp_times[i] - time))
        idx_list.append(time_idx)
    return idx_list

def p_times(arrays_to_compare, channels = 'all'):
    """ 
    calculate p values of differences between the pre- and post-training ERPs
    currently using ind samples t-test but should reconsider this...

    arrays_to_compare: a list of 2 arrays to compare. Like [test_pre, test_post]
    returns: a list of p values, one p value for each time point
    """
    p_values = []
    if channels == 'all':
        print('Calculating p-value over mean of all channels')
        for timepoint in range(0, arrays_to_compare[0].shape[2]):
            
            array1 = arrays_to_compare[0][:, 0:64]
            array2 = arrays_to_compare[1][:, 0:64]
            res = ttest_ind(array1.mean(axis = 1)[:, timepoint], array2.mean(axis = 1)[:, timepoint])
            p_values.append(res.pvalue)

    
    elif type(channels) == list:
        print(f'Calculating p-value over {channels}')
        for timepoint in range(0, arrays_to_compare[0].shape[2]):
            p_ch_idx = ch_index(channels)
            array1 = arrays_to_compare[0][:, p_ch_idx]
            array2 = arrays_to_compare[1][:, p_ch_idx]
            res = ttest_ind(array1.mean(axis = 1)[:, timepoint], array2.mean(axis = 1)[:, timepoint])
            p_values.append(res.pvalue)

    else:
        print('Valid channel arguments: type list')
        exit

    return p_values

"""
KS test to determine whether the data is normally distributed. 
Takes an array of shape [n_participants, n_timepoints] and determines whether distribution 
is normal at each timepoint.
Returns significance level of KS test averaged over all timepoints
"""
def gaussian_test(array):
    p_values = []
    for timepoint in range(0, array.shape[1]):
        res = kstest(array[:,timepoint], 'norm')
        p_values.append(res.pvalue)

    p_values = np.array(p_values)
    significance = p_values.mean()

    if significance > 0.05: 
        print("Distribution is normal. p = ", significance)
    elif significance < 0.04:
        print("Distribution is not normal. p = ", significance)

    return significance

"""
Adaptation of scipy implmentation but comparing one sample with an expected mean of 0
Only one input array needed
returns the result of the wilcoxon test
"""
def wilcoxon_1samp(array):
    pop_mean = np.zeros_like(array)
    res = wilcoxon(array, pop_mean)
    return res

""" 
calculate p values of differences between the pre- and post-training ERPs
uses the within-subjects t-test

array: preprocessed array containing the average ERP (post minus pre) for each subject
returns: a list of p values, one p value for each time point
"""

def p_times_1sample(array, channels = 'all'):

    if channels == 'all':
        print('Calculating p-value over mean of all channels')
        array_ch_mean = array.mean(axis = 1)
    elif type(channels) == list:
        print(f'Calculating p-value over {channels}')
        p_ch_idx = ch_index(channels)
        array = array[:, p_ch_idx]
        array_ch_mean = array.mean(axis = 1)
    else:
        print('Valid channel arguments: type list')
        exit

    #test normality
    ks = gaussian_test(array_ch_mean)

    p_values = []
   # ks = 1
    for timepoint in range(0, array.shape[2]):
        if ks > 0.05:
            res = ttest_1samp(array_ch_mean[:, timepoint], popmean = 0)       
        elif ks < 0.05:
            res = wilcoxon_1samp(array_ch_mean[:, timepoint])
        p_values.append(res.pvalue)
    
    return p_values

"""
This function is used to process the data so that it's ready for the 1 sample t-test. Takes the 
subject averages for each condition, and subtracts them
Returns an array of evoked data where the first dim is the number of subjects (supposedly)
"""
def find_diff_sa(evoked_list1, evoked_list2):
    diff_evoked_list = [evoked1 - evoked2 for evoked1, evoked2 in zip(evoked_list1, evoked_list2)]
    diff_evoked_sa = np.stack(diff_evoked_list)
    return diff_evoked_sa

""" 
calculate p values of differences between the pre- and post-training ERPs
currently using ind samples t-test but should reconsider this...

arrays_to_compare: a list of 2 arrays to compare. Like [test_pre, test_post]
returns: a list of p values the , one p value for each time point
"""


def p_chs(arrays_to_compare, time_idx):
    p_values = []
 
    for channel in range(0, 64):
        array1 = arrays_to_compare[0][:, channel, time_idx]
        array2 = arrays_to_compare[1][:, channel, time_idx]
        res = ttest_ind(array1, array2)
        p_values.append(res.pvalue)
        
    return p_values

""" 
Process the p-values so that when they're plotted as a topomap, the small values (i.e. the most significant) are plotted as red
Also anything > 0.05 becomes 0 
"""

def scale_p_channels(p_values, threshold = 0.95):
    scaled_values = [1-x for x in p_values]
    scaled_values = [0 if x < threshold else x for x in scaled_values]


    return scaled_values

def compute_power(epochs, tmin = 0, tmax = 0.25, bands=['delta', 'theta', 'alpha', 'beta', 'gamma', 'all'], method = 'welch'):
    """
    Returns a DataFrame with power computed over each frequency band for given epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs for which to compute the PSD and power.
    bands : list of str, optional
        List of frequency bands to compute power for. Default is ['delta', 'alpha'].
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame where each column represents the power in a different frequency band.
    """

    freqbands = {'delta': [0.5, 4], 
                 'theta': [4, 8],
                 'alpha': [8, 12],
                 'beta': [12, 30],
                 'gamma': [30, 45],
                 'all': [0.5, 45]
                }

    
    power_dict = {}
    for key in bands:
        if key not in freqbands:
            continue  
        fmin, fmax = freqbands[key]

        psd = mne.Epochs.compute_psd(epochs, 
                                     method = method,
                                     fmin=fmin, 
                                     fmax=fmax, 
                                     tmin = tmin, 
                                     tmax = tmax)
        
        psd_ave_64 = psd.average() #average over epochs
        psd_ave = np.mean(psd_ave_64.get_data(), axis = 0) #average over channel

        #integrate PSD
        power = np.trapz(psd_ave)

        # save PSD
        power_dict[key] = power


    df = pd.DataFrame([power_dict])

    return df


def load_evoked_epochs(subjects_to_process, task):

    evoked_dir = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_{task}/{task}_ERP_data'
    epochs_dir = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_{task}/{task}_epochs_data'

    """
    Loads the epochs and evoked .fif files and organizes them into lists to use for plotting and analysis
    subjects_to_process: list of subjects where each element is a string. e.g. ['01', '02']

    ---
    Returns concatenated epochs and evoked lists: concat_epochs_pre, concat_evoked_pre, concat_epochs_post, concat_evoked_post
    """
    evoked_list_pre = []
    epochs_list_pre = []
    evoked_list_post = []
    epochs_list_post = []

    #subject averages
    #epochs_list_pre_sa =[]

    #for file in sorted(os.listdir(evoked_dir)):
    assert isinstance (subjects_to_process, list)
    for subject in subjects_to_process:
        print('Processing subject: ', subject)

        file_evokeds_pre = glob.glob(os.path.join(evoked_dir, f'{task}_ERP_pre_{subject}.fif'))[0]
        file_epochs_pre = glob.glob(os.path.join(epochs_dir, f'{task}_epochs_pre_{subject}.fif'))[0]
    
        evoked_pre = mne.read_evokeds(file_evokeds_pre)[0]
        evoked_list_pre.append(evoked_pre)
        epochs_pre = mne.read_epochs(file_epochs_pre)
        epochs_list_pre.append(epochs_pre)

        file_evokeds_post = glob.glob(os.path.join(evoked_dir, f'{task}_ERP_post_{subject}.fif'))[0]
        file_epochs_post = glob.glob(os.path.join(epochs_dir, f'{task}_epochs_post_{subject}.fif'))[0]
    
        evoked_post = mne.read_evokeds(file_evokeds_post)[0]
        evoked_list_post.append(evoked_post)
        epochs_post = mne.read_epochs(file_epochs_post)
        epochs_list_post.append(epochs_post)


    concat_epochs_pre = mne.concatenate_epochs(epochs_list_pre)
    concat_evoked_pre = mne.combine_evoked(evoked_list_pre, weights = 'equal')

    concat_epochs_post = mne.concatenate_epochs(epochs_list_post)
    concat_evoked_post = mne.combine_evoked(evoked_list_post, weights = 'equal')
    
    return (concat_epochs_pre, concat_evoked_pre, concat_epochs_post, concat_evoked_post)

def load_epochs_bysubject(subjects_to_process, task):


    """
    Loads the epochs and evoked .fif files and organizes them into a dataframe to use for plotting and analysis
    subjects_to_process: list of subjects where each element is a string. e.g. ['01', '02']


    ---
    Returns a dataframe with columns 'subject', 'period', 'musician', and 'epochs'.
        each row of ['epochs'] is an array of shape n_channels x n_timepoints, and is the average of all epochs from one subject
    """
    
    epochs_dir = f'/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/analysis_{task}/{task}_epochs_data'
    epochs_df = pd.DataFrame(columns = ['subject', 'period', 'musician', 'epochs'])
    
    good_listen_subjects, good_motor_subjects, musicians, nonmusicians = load_subject_lists()

    #subject averages
    #epochs_list_pre_sa =[]

    #for file in sorted(os.listdir(evoked_dir)):
    assert isinstance (subjects_to_process, list)

    for subject in subjects_to_process:
        print('Processing subject: ', subject)

        if subject in musicians: 
            musician = 1
        else: 
            musician = 0
        for period in ['pre', 'post']:
            file_epochs_pre = glob.glob(os.path.join(epochs_dir, f'{task}_epochs_{period}_{subject}.fif'))[0]
            epochs_sub = mne.read_epochs(file_epochs_pre)
            epochs_sub = np.mean(epochs_sub.get_data()[:, :64, :], axis = 0) #get only the eeg channels and average all trials per subject
            #epochs_sub = epochs_sub[np.newaxis, :, :]

            df_sub = pd.DataFrame({
                'subject': subject,
                'period' : period,
                'musician' : musician,
                'epochs': [epochs_sub]
            })
            epochs_df = pd.concat([epochs_df, df_sub])


    epochs_df.reset_index(drop=True, inplace=True)
    return (epochs_df)


