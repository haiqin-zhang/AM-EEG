"""
Utils for mTRF analyses
"""

"""
Cuts the data into equally-sized fragments for mTRFpy. Takes an array and the number of segments 
"""
#obselete version that only works on 1d arrays
def segment(arr, n_segments):
    segment_size = len(arr) // n_segments  # Calculate the size of each segment
    segments = [arr[i * segment_size : (i + 1) * segment_size] for i in range(n_segments)]  # Slice the array into 10 segments
    return segments

"""
def segment(arr, n_segments, axis=0)
    # Determine the length of the array along the specified axis
    segment_size = arr.shape[axis] // n_segments

    # Slice the array along the chosen axis
    if axis == 0:  # Segment along rows
        segments = [arr[i * segment_size : (i + 1) * segment_size, :] for i in range(n_segments)]
    elif axis == 1:  # Segment along columns
        segments = [arr[:, i * segment_size : (i + 1) * segment_size] for i in range(n_segments)]
    else:
        raise ValueError("Invalid axis. Axis must be 0 (rows) or 1 (columns).")

    return segments"""