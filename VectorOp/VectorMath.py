import numpy as np
'''
returns the corners of the bounding box in the form "top left corner x, top left corner y, bottom right corner x, bottom right corner y"
'''
def vectors_bounding_box(vectors):
    small_x = vectors[0][0]
    big_x = vectors[0][0]
    small_y = vectors[0][1]
    big_y = vectors[0][1]

    for i in range(1, len(vectors)):
        iter_xy = vectors[i]
        if iter_xy[0] > big_x:
            big_x = iter_xy[0]
        if iter_xy[0] < small_x:
            small_x = iter_xy[0]
        if iter_xy[1] < small_y:
            small_y = iter_xy[1]
        if iter_xy[1] > big_y:
            big_y = iter_xy[1]
    return np.array([small_x, small_y, big_x, big_y])
