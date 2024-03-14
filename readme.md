Repository for EEG analysis in the audiomotor piano project.
All analyses are done in the eelbrain-cnsp environment, requirements in .txt file.

Trigger channels:
Trigger ID - soundcard channel (as shown on Ableton and soundcard ports)

Output 1 always goes to the speaker or headphones (what the participant hears)
65282 = output 2 - always records the MIDI keyboard keystrokes
65284 = output 3 
65288 = output 4 
65290 = not sure...
65296 = output 5


not sure what's up with 65890, doesn't match any of the triggers

Outputs 3+ are used for different things depending on the part of the experiment:

Passive listening:
Output 2 - each individual note played
Output 5 - beginning and end of trial

Motor tests:
Output 2 - keystrokes
Output 5 - beginning and end of trial 

Additional ableton channels with volume set to 0 (so nothing goes to soundcard) to record the corresponding audio in the Normal and inv mappings can be exported after the experiment

Error tests:
(mapping codes: inv = inverted, shinv = shifted inverted, norm = normal piano keyboard)
Output 3 - trig-inv
Output 4 - trig-shinv
Output 5 - beginning and end of trial; during the trial it marks trig-norm

Naming convention in analysis files:
t_trigname = all the triggers with the same event type straight from MNE
tc_trigname = cleaned list of triggers that only takes the first event of each trigger