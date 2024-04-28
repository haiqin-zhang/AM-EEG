import numpy as np
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind
import pickle

#get channel names
with open('../utils/ch_names.pkl', 'rb') as file:
    ch_names_all = pickle.load(file)

ch_names_72 = ch_names_all[0:72]

"""
find index of a channel in the data array given the channel name. Relies on the preloaded list of channels.

Takes a list of channel names, e.g. ['Cz']
Returns a list of indices
"""

def ch_index(ch_list): 
    ch_indices = [ch_names_72.index(item) if item in ch_names_72 else None for item in ch_list]
    return ch_indices


""" 
calculate p values of differences between the pre- and post-training ERPs
currently using ind samples t-test but should reconsider this...

arrays_to_compare: a list of 2 arrays to compare. Like [test_pre, test_post]
returns: a list of p values the , one p value for each time point
"""


def p_times(arrays_to_compare, channels = 'all'):
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

