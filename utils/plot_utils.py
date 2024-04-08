"""
Functions for visualizing analyses
"""

import matplotlib.pyplot as plt

"""
Plots the sections in time corresponding to each playing mode: inv, shinv and norm. Each is shaded a different colour.
section_list: the list of start and end times of a certain type of section. Found using the function find_sections in pp_utils

example: plot_sections(inv_sections)
"""
def plot_sections(section_list, downfreq = 32):
    plt.figure(figsize=(20,10))

    y1 = 0
    y2 = -1
    colors = ['green', 'orange', 'red']
    colour_idx = 0

    for sections in section_list:
        y = [y1, y1, y2, y2]
        sections = sections/downfreq 
        for i in range(sections.shape[0]):
            x = [sections[i][0], sections[i][1], sections[i][1], sections[i][0]]
            plt.fill(x, y, color=colors[colour_idx], alpha=0.2)
        colour_idx +=1
        y1 -=1
        y2-=1

    
    plt.xlabel('Time (s)')
    plt.ylabel('Playing mode')
    plt.show()
