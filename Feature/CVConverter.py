import numpy as np

def kp_to_tuple(kp):
    return (kp.pt[0], kp.pt[1], kp.size)

def kp_to_tuple_int(kp):
    return (int(kp.pt[0]), int(kp.pt[1]), int(kp.size))

def kp_to_np(kp):
    return np.array([kp.pt[0], kp.pt[1], kp.size])
