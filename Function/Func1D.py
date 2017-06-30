import numpy as np

def norm_by_area(arr):
    return arr/float(np.trapz(arr))
