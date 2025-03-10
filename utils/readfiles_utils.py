import pickle

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)
    

def onsets_pitch_from_midi(file):
    """ 
    Extracts the midi values from a file and returns them in a list
    file: path of the MIDI file to be read
    """
    midi_file = mido.MidiFile(file)
    note_onsets = []

    for msg in midi_file:
        if msg.type == "note_on" and msg.velocity > 0:  
            note_onsets.append((msg.note))

    return note_onsets
