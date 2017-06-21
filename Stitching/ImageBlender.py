import cv2
from PIL import Image
import numpy as np

'''
def feather_blend(image1, image2, window_size):
    thresh_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    thresh_image1[thresh_image1 > 0] = 255.0
    thresh_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    thresh_image2[thresh_image2 > 0] = 255.0


    window = (window_size, window_size)
    mag_resp_image1 = cv2.blur(thresh_image1, window)

    mag_resp_image2 = cv2.blur(thresh_image2, window)
    Image.fromarray(mag_resp_image1).show()
    Image.fromarray(mag_resp_image2).show()

    proportion_resp_image1 = np.zeros(thresh_image1.shape)
    proportion_resp_image2 = np.zeros(thresh_image2.shape)

    proportion_resp_image1 = mag_resp_image1/(mag_resp_image1 + mag_resp_image2)
    proportion_resp_image2 = mag_resp_image2/(mag_resp_image2 + mag_resp_image1)

    Image.fromarray(proportion_resp_image1).show()
    Image.fromarray(proportion_resp_image2).show()
'''

def feather_blend(blend_image, base_image, window_size):
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

    out_image = np.uint8(weighted_blend_image + weighted_base_image)#np.uint8(blend_image * blend_response_image + base_image * base_response_image)

    #Image.fromarray(out_image).show()
    return out_image
