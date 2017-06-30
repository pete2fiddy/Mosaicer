import numpy as np
from math import pi

'''takes a small window and creates an un-normalized HOG histogram for the patch'''
def HOG_window(phase_window, mags_window, num_bins):
    hist = np.zeros((num_bins))
    theta_per_index = pi/float(num_bins)
    flat_phases = phase_window.flatten()
    flat_mags = mags_window.flatten()
    for i in range(0, flat_phases.shape[0]):
        lower_hist_index = int(flat_phases[i]/theta_per_index)
        upper_hist_index = lower_hist_index + 1 if lower_hist_index < hist.shape[0]-1 else 0
        proportion_to_lower_index = (flat_phases[i] - (theta_per_index * lower_hist_index))/theta_per_index
        proportion_to_upper_index = 1.0 - proportion_to_lower_index
        hist[lower_hist_index] += proportion_to_lower_index * float(flat_mags[i])
        hist[upper_hist_index] += proportion_to_upper_index * float(flat_mags[i])
    return hist
