Repository for EEG analysis in the audiomotor piano project.
All analyses are done in the eelbrain-cnsp environment, requirements in .txt file.

Trigger channels:
Trigger ID - soundcard channel (as shown on Ableton and soundcard ports)

Output 1 always goes to the speaker (what the participant hears)
65282 = output 2 - always records the MIDI keyboard keystrokes
65284 = output 3 
65288 = output 4 
not sure what's up with 65890 (maybe output 5 but I didn't really test it)

Outputs 3+ are used for different things depending on the part of the experiment

Motor tests (post only):
Output 3 - trig-mute
Output 4 - trig-unmute
Additional ableton channels with volume set to 0 (so nothing goes to soundcard) to record the corresponding audio in the normal and inv mappings to be exported later

Error tests:
(mapping codes: inv = inverted, shinv = shifted inverted)
Output 3 - trig-inv
Output 4 - trig-shinv


t_trigname = all the triggers with the same event type straight from MNE
tc_trigname = cleaned list of triggers that only takes the first event of each trigger