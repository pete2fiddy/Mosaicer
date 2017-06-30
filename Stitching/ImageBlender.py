import cv2
from PIL import Image
import numpy as np
from Feature.AlignSolve import AlignSolve


def feather_blend(blend_image, base_image, blend_params):
    window_size = blend_params["window_size"]
    blend_thresh_image = cv2.cvtColor(blend_image, cv2.COLOR_RGB2GRAY)
    blend_thresh_image[blend_thresh_image > 0] = 255.0
    base_thresh_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)
    base_thresh_image[base_thresh_image > 0] = 255.0

    window = (window_size, window_size)
    blend_response_image = cv2.blur(blend_thresh_image, window)
    blend_response_image[blend_response_image < 254] = 0
    blend_response_image = cv2.blur(blend_response_image, window)
    base_response_image = 255.0-blend_response_image


    blend_response_image = np.float32(blend_response_image/255.0)
    '''ISSUE:
    near corners, this gets rid of weights between the blend and base image and prefer 100% weighting toward one rather than balancing
    both. Is still better than the fuzzy edges from before, however'''
    blend_response_image[base_thresh_image == 0] = 1.0
    base_response_image = np.float32(base_response_image/255.0)

    weighted_blend_image = blend_response_image[:, :, np.newaxis] * blend_image


    weighted_base_image = base_response_image[:,:,np.newaxis] * base_image

    out_image = np.uint8(weighted_blend_image + weighted_base_image)
    return out_image

def paste_blend(blend_image, base_image, blend_params = None):
    thresh_blend_image = cv2.cvtColor(blend_image, cv2.COLOR_RGB2GRAY)
    thresh_base_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)

    thresh_blend_image[thresh_blend_image > 0] = 1.0
    thresh_base_image[thresh_base_image > 0] = 1.0


    ignore_space = np.logical_not(np.logical_and(thresh_blend_image, thresh_base_image)).astype(np.float32)

    out_image = blend_image.copy().astype(np.float32)
    out_image += ignore_space[:,:,np.newaxis] * base_image
    return np.uint8(out_image)
