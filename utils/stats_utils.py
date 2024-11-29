import numpy as np
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind, kstest, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import pickle
import mne
import pandas as pd

import glob
import os
from update_sub_lists import*
from pp_utils import *


def gaussian_test(array, axis = 0):
    """
    KS test to determine whether the data is normally distributed. 
    Takes an array of shape [n_participants, n_timepoints] and determines whether distribution 
    is normal at each timepoint.
    axis: which axis of the array to test the array over
    ---
    Returns:
    significance: p-value of KS test averaged over all timepoints
    """
    n_points = array.shape[axis]
    p_values = []
    
    for timepoint in range(0, n_points):
        res = kstest(array[:,timepoint], 'norm')
        p_values.append(res.pvalue)

    p_values = np.array(p_values)
    print(f'testing gaussianity over {n_points} points')
    significance = p_values.mean()

    if significance > 0.05: 
        print("Distribution is normal. p = ", significance)
    elif significance < 0.04:
        print("Distribution is not normal. p = ", significance)

    return significance

def t_within_points(array, lim1 = None, lim2 = None):
    """ 
    compare within-subjects samples along an axis of points (e.g. times or freqs)
    uses the 1-sample t-test if normal and wilcoxon otherwise

    array: array of shape n_subs x n_points
    subaxis: which dim contains the subjects
    pointaxis: which dim contains the 
    lim1, lim2: indices of the points over which we want to compute differences (e.g. from 4-30 Hz). 
        Find the indices beforehand using index_custom()
        (prevents loss of power through multiple comparisons) 
    ---
    returns: (both of shape n_points)
    t: a list of t statistics
    p: a listp values, one p value for each time point
    """
    n_points = array.shape[1]
    ks = gaussian_test(array, axis = 0)


    test_stats = []
    p_values = []

    if ks > 0.05:
        print('using 1samp ttest')
        test_function = lambda array, point: ttest_1samp(array[:, point])
    elif ks < 0.05:
        print('using wilcoxon test')
        test_function = lambda array, point: wilcoxon(array[:, point])

    for point in range(n_points):
        res = test_function(array, point)
        test_stats.append(res.statistic)
        p_values.append(res.pvalue)
    

    #FDR CORRECTION
    if lim1 != None:
        assert lim2 != None, 'define both lim1 and lim 2, otherwise both should be None'

        #correct order of lims in case lim2 > lim 1
        upper_lim = np.max([lim1, lim2])
        lower_lim = np.min([lim1, lim2])

        p_values_lim = p_values[lower_lim:upper_lim]
        p_values_corrected = fdrcorrection(p_values_lim)[1]
        p_values_corrected = np.pad(p_values_corrected, (lower_lim,len(p_values)-upper_lim), mode='constant', constant_values=1)


        print(f'fdr correction over {len(p_values_lim)} points')
    else:
        p_values_corrected = fdrcorrection(p_values)[1]
        print('fdr correction over all p values')
        

    return test_stats, p_values_corrected




def jackknife_1samp(data_point, test):
    """
    estimate distribution of p values (also test statistics)
    data_freq: data of shape n_subs, already indexed for the freq or timepoint of interest
    test: which test to do. Usually wilcoxon

    ---
    returns:
    p_vals: distribution of p values
    ---
    """
    n_subs = data_point.shape[0]

    t_stat_jk = []
    p_val_jk = []
    
    for sub in range(n_subs):
        data_resampled = np.delete(data_point, sub)
        res = test(data_resampled)
        p_val_jk.append(res.pvalue)
        t_stat_jk.append(res.statistic)

    return t_stat_jk, p_val_jk
        
    


