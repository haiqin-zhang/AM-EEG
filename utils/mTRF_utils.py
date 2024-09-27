"""
Utils for mTRF analyses
"""

import numpy as np


def segment(arr, n_segments):
    """
    Cuts the data into equally-sized fragments for mTRFpy. Takes an array and the number of segments 
    Works on arrays of different dimensions but uses slices it along len(arr) i.e. along the first dimension or horizontally
    """
    segment_size = len(arr) // n_segments  # Calculate the size of each segment
    segments = [arr[i * segment_size : (i + 1) * segment_size] for i in range(n_segments)]  # Slice the array into 10 segments
    return segments

def normalize_responses(responses):
    """Function that normalizes the EEG data
    Data is normalized by subtracting the
    global mean and dividing by the global
    standard deviation of the data to preserve
    relative amplitudes between channels
   
    Args:
    responses: list of numpy arrays, each array
    contains the EEG data for one trial. The
    data must be organized as n_samples x n_channels
    """

    # Check dimensions
    if isinstance(responses, list):
        n_rows, n_cols = responses[0].shape
        if n_rows < n_cols:
            raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')
        responses_concatenated = np.concatenate(responses, axis=0).flatten()
    else:
        raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')

    global_mean = np.mean(responses_concatenated)
    global_std = np.std(responses_concatenated)
    responses = [(response - global_mean) / global_std for response in responses]

    return responses


def normalize_stimuli(stimuli):
    """Function that normalizes the stimuli data
    Data is normalized by dividing each feature
    by the maximum value of that feature

    Args:
    stimuli: list of numpy arrays, each array
    contains the stimuli data for one trial. The
    data must be organized as n_samples x n_features
    """

    # Check dimensions
    if isinstance(stimuli, list):
        n_rows, n_cols = stimuli[0].shape
        if n_rows < n_cols:
            raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')  
        stimuli_concatenated = np.concatenate(stimuli, axis=0)
    else:
        raise Exception('Data should be a list of numpy arrays with dimesions n_samples x n_channels')

    feats_max = np.max(stimuli_concatenated, axis=0)
    stimuli = [stimulus / feats_max for stimulus in stimuli]

    return stimuli
