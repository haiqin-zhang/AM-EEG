"""
Utils for mTRF analyses
"""

"""
Cuts the data into equally-sized fragments for mTRFpy. Takes an array and the number of segments 
"""
def segment(arr, n_segments):
    segment_size = len(arr) // n_segments  # Calculate the size of each segment
    segments = [arr[i * segment_size : (i + 1) * segment_size] for i in range(10)]  # Slice the array into 10 segments
    return segments