"""
Some functions for analysis. Updated every once in a while maybe...?
"""


import numpy as np
import mne


"""
Filters the trigger signals so that only the first value of each group of triggers that is close in time is retained. 
Necessary because the triggers are analog so each real trigger results in a string of triggers being detected.

Trig array: the array straight out of mne.find_events
Threshold: the minimum difference in the first column between the previous and current row for the current row to be retained 
""" 
def clean_triggers(trig_array, threshold = 100):
    #setting threshold for how much time has to pass between each trigger
    diff_time = np.diff(trig_array[:, 0])
    indices = np.where(diff_time > threshold)[0]
    indices = np.concatenate(([0], indices + 1))
    return trig_array[indices]


"""
Concatenates different sections of the EEG experiment that are OF THE SAME LENGTH
(It does the same thing as Epochs sort of, but you end up with a raw object instead of an epoch object so you can work with it further...)
Useful for gathering all the recordings of muted and unmuted sections of the motor experiment. 

raw: the original raw file straight out of mne.io.read_raw_bdf
events: the events array of interest. Take only the first column, like tc_mute[:,0]
segment_dur: duration of the segment. For muted segments it's 30 seconds, for unmuted it's 10


Note: the time axis will still be continuous even if the data has been chopped up
"""
def concat_uniform(raw, events, segment_dur):
    segments = []
    for time in events[:,0]:

        segment_start = time/fs #because crop() uses seconds and not samples
        segment_end = segment_start+segment_dur

        #make sure the max length of eeg is not exceeded
        if segment_end > raw.times.max():
            segment_end = raw.times.max()

        segment = raw.copy().crop(tmin = segment_start, tmax = segment_end)
        segments.append(segment)

    return mne.io.concatenate_raws(segments)


""" 
Concatenates different sections of the EEG experiment that are OF DIFFERENT LENGTHS
Useful for gathering all the recordings of the inv and norm mapping segments of error trials

raw: the original raw file straight out of mne.io.read_raw_bdf
events1: trigger array marking beginning of the segment of interest. Take only first column. Ex. trig_inv[:,0]
events2: trigger array marking beginning of other segments (and therefore the end of the segment of interest)
"""
def concat_nonuniform(raw, events1, events2):
    segments = []
    for time in events1[:,0]:

        segment_start = time/fs #because crop() uses seconds and not samples
        remaining_trigs =  [x for x in events2[:,0] if x > time]
        if len(remaining_trigs) > 0: 
            segment_end = remaining_trigs[0]/fs

        #make sure the max length of eeg is not exceeded
        else:
            segment_end = raw.times.max()

        segment = raw.copy().crop(tmin = segment_start, tmax = segment_end)
        segments.append(segment)
        
    return mne.io.concatenate_raws(segments)
