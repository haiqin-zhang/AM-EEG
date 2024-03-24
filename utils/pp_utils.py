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
def clean_triggers(trig_array, threshold=100):
    cleaned_triggers = []
    prev_trigger_time = trig_array[0, 0]  # Initialize with the first trigger time
    cleaned_triggers.append(trig_array[0])  # Retain the first trigger
    
    for trigger in trig_array[1:]:
        trigger_time = trigger[0]
        if trigger_time - prev_trigger_time > threshold:
            cleaned_triggers.append(trigger)
            prev_trigger_time = trigger_time
    
    return np.array(cleaned_triggers)

"Preserves short events during downsampling"
def discretize(arr, final_length, downfreq_factor = 32):
    n_bins = int(len(arr)//downfreq_factor)

    if len(arr) % downfreq_factor != 0:
        remainder = downfreq_factor - (len(arr) % downfreq_factor)
        arr = np.append(arr, [0] * int(remainder))
        n_bins +=1

    arr_reshaped = arr.reshape(n_bins, downfreq_factor)
    discretized_arr = np.any(arr_reshaped, axis=1).astype(int)

    discretized_arr = discretized_arr[0:final_length]

    return discretized_arr

""" 
Sorts the events found by mne into events from different soundcard channels.

events: array of all events after cleaning by clean_triggers

Returns separate arrays with only one event type each. 
Also returns a list of start times for each type of trial (listening, motor, error)

Example:

events_2, events_3, events_4, events_5, trial_starts = sort_events(events)
"""
def sort_events(events, clean = True):
    #assert 65282 and 65284 and 65288 and 65296 in events[:,2], "Not all trig categories are present"

    if 65282 and 65284 and 65288 and 65296 in events[:,2]:
        print("All event types present")    
    else:
        print("Some event types missing. Check data.")

    if clean == True:
        events_2 = clean_triggers(events[events[:,2] == 65282]) #t2 - keystrokes
        events_3 = clean_triggers(events[events[:,2] == 65284]) #t3
        events_4 = clean_triggers(events[events[:,2] == 65288]) #t4
        events_5 = clean_triggers(events[events[:,2] == 65296]) #t5 (it's also used for trials, so there should be two far apart at the beginning)

        #get only the start triggers that are at least 11min 10 secs (670 s) mins apart
        #motor and error trials are exactly 10 mins long. Passive listening is 11:05 mins.
        trial_starts = clean_triggers(events[events[:,2] == 65296], threshold = 1372160) 
    
    else:
        events_2 = events[events[:,2] == 65282]
        events_3 = events[events[:,2] == 65284]
        events_4 = events[events[:,2] == 65288]
        events_5 = events[events[:,2] == 65296]
        trial_starts = events[events[:,2] == 65296]

    return events_2, events_3, events_4, events_5, trial_starts


""" 
Finds timeframes of each playing mode in the error trial (inv, shinv, or norm)
raw: eeg data
section_triggers: the mode of interest
t_modeswitch: events, defined beforehand. All the mode-related triggers

Example: 
find_sections(raw, t_shinv, t_modeswitch)
"""
def find_sections(raw, section_trigggers, mode_triggers):
    section_times = []
    for segment_start in section_trigggers[:,0]:

        #segment_start = time #because crop() uses seconds and not samples
        remaining_trigs =  [x for x in mode_triggers[:,0] if x > segment_start]
        if len(remaining_trigs) > 0: 
            segment_end = remaining_trigs[0]

        #make sure the max length of eeg is not exceeded
        else:
            segment_end = raw.times.max()

        section_times.append([segment_start, segment_end])

    return np.array(section_times)


""" 
Finds the keystrokes that fall into a certain condition.
raw: EEG data
t_keystrokes: all keystroke events
timeframes: the times of the condition you want, found with find_sections

Example: find_keystrokes(raw, t_keystrokes, norm_times)
"""
def find_keystrokes(raw, t_keystrokes, timeframes):
    keystrokes = t_keystrokes[:,0]
    filt_keystrokes = []
    for segment in timeframes:
        start = segment[0]
        end = segment[1]

        condition = (keystrokes >= start) & (keystrokes <= end)
        filt_keystrokes.extend(keystrokes[condition])

    indices = np.where(np.isin(t_keystrokes[:, 0], filt_keystrokes))[0]
    filt_keystrokes_events = t_keystrokes[indices]
    return filt_keystrokes_events










#-------------------------------------------------------------------
# STUFF BELOW NOT IN USE
#-------------------------------------------------------------------

"""
Concatenates different sections of the EEG experiment that are OF THE SAME LENGTH
(It does the same thing as Epochs sort of, but you end up with a raw object instead of an epoch object so you can work with it further...)
Useful for gathering all the recordings of muted and unmuted sections of the motor experiment. 

raw: the original raw file straight out of mne.io.read_raw_bdf
events: the events array of interest. Take only the first column, like tc_mute[:,0]
segment_dur: duration of the segment. For muted segments it's 30 seconds, for unmuted it's 10


Note: the time axis will still be continuous even if the data has been chopped up
"""
def concat_uniform(raw, events, segment_dur, fs):
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
def concat_nonuniform(raw, events1, events2, fs):
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
