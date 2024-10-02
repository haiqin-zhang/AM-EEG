## Participant inclusion

all_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

good_listen_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
good_motor_subjects = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'] #double check 21


nonmusicians = ['01', '03', '04', '05', '08', '09', '10', '11', '16', '19', '20']
musicians = ['02', '06', '07', '12', '13', '14', '15', '17', '18', '21']

These lists are saved as .mat files under utils and can be loaded later
---
## Participant-specific preprocessing notes

### New info for participants 14+: MIDI triggers added
**Preprocess using pp_manual_new and pp_prepost_new**: these files allow keypresses to be aligned to MIDI triggers 

Participant 13: special case, MIDI triggers only use one channel but there are bugs  

Participants 14+ have both audio and MIDI triggers (MIDI triggers using all channels 9-16)  
---

### Info for participants 1-12

Contains only triggers related to Ableton audio 

### Preprocess using pp_manual and pp_prepost
pp_manual is for recordings with bugs so that you can preprocess each segment (listen, motor, error) individually 
pp_prepost does all pre- and post-training processing in batch

Trigger channels:
Trigger ID - soundcard channel (as shown on Ableton and soundcard ports)

Output 1 always goes to the speaker or headphones (what the participant hears)
65282 = output 2 - always records the MIDI keyboard keystrokes  
65284 = output 3  
65288 = output 4  
65290 = not sure...  (doesn't match any of the triggers)
65296 = output 5  

These outputs change with the new MIDI trigger box! 2, 4, 8, 16 are 2, 3, 4, 5 respectively. The MIDI triggers where all the channels light up is 65282. 


Outputs are used for different things depending on the part of the experiment:


Passive listening:  
Output 2 - each individual note played  
Output 5 - beginning and end of trial  
 
Motor tests:  
Output 2 - keystrokes  
Output 5 - beginning and end of trial   

Additional ableton channels with volume set to 0 (so nothing goes to soundcard) to record the corresponding audio in the normal and inv mappings can be exported after the experiment.

Error tests:
(mapping codes: inv = inverted, shinv = shifted inverted, norm = normal piano keyboard)
Output 3 - trig-inv
Output 4 - trig-shinv
Output 5 - beginning and end of trial; during the trial it marks trig-norm

Naming convention in analysis files:
t_trigname = all the triggers with the same event type straight from MNE
tc_trigname = cleaned list of triggers that only takes the first event of each trigger