from Toolbox.NamedArgs import NamedArgs
import numpy as np
from PIL import Image
import cv2
from math import log

class MosaicMaker:
    '''creates a single mosaic image out of the two inputted images using the inputted parameters, feature selection types, etc.'''
    def __init__(self, image1, image2, match_type, align_solve_type, mask = None):
        self.image1 = image1
        self.image2 = image2
        self.mask = mask

        self.match_type = match_type(self.image1, self.image2, self.mask)
        self.align_solve_type = align_solve_type


    '''to prevent "bad mosaic transforms," may be possible to add some constraints that can be determined if a transformed image is unreasonably squashed'''
    def create_mosaic(self, ransac_params, blend_func_and_params, return_stitches = False):

        stitch1, stitch2, shift = self.create_stitch_images(ransac_params)

        '''may not be necessary, stitch1 and stitch2 may be able to be "and'ed" without thresholding'''
        thresh_stitch1 = cv2.cvtColor(stitch1, cv2.COLOR_RGB2GRAY)
        thresh_stitch1[thresh_stitch1 > 0] = 1.0
        thresh_stitch1 = thresh_stitch1.astype(np.bool)
        thresh_stitch2 = cv2.cvtColor(stitch2, cv2.COLOR_RGB2GRAY)
        thresh_stitch2[thresh_stitch2 > 0] = 1.0
        thresh_stitch2 = thresh_stitch2.astype(np.bool)


        stitch_union = np.uint8(255*np.logical_and(thresh_stitch1, thresh_stitch2).astype(np.int))

        union_image1 = cv2.cvtColor(stitch1, cv2.COLOR_RGB2GRAY)
        union_image2 = cv2.cvtColor(stitch2, cv2.COLOR_RGB2GRAY)
        union_image1[stitch_union < 1] = 0
        union_image2[stitch_union < 1] = 0
        union_image1 = union_image1.astype(np.float32)/255.0
        union_image2 = union_image2.astype(np.float32)/255.0

        mean1 = np.average(union_image1, axis = (0,1)) * (union_image1.shape[0] * union_image1.shape[1]) / float(np.count_nonzero(stitch_union))
        mean2 = np.average(union_image2, axis = (0,1)) * (union_image2.shape[0] * union_image2.shape[1]) / float(np.count_nonzero(stitch_union))
        stitch1 = stitch1.astype(np.float32)
        stitch2 = stitch2.astype(np.float32)

        gamma_adjust1 = log(mean2, mean1)
        stitch1 = 255*(stitch1/255.0)**(gamma_adjust1)


        '''image1_brightness_add = mean2 - mean1
        print('image1 brightness add: ', image1_brightness_add)
        image1_brightness_multiplier = mean2/mean1
        print("image1 brightness multiplier: ", image1_brightness_multiplier)


        stitch1[union_image1 > 1] += image1_brightness_add
        #stitch2[union_image1 > 1] -= image1_brightness_add/2.0'''

        stitch1 = np.uint8(stitch1)
        blended_image = blend_func_and_params.inst(stitch1, stitch2, blend_func_and_params)
        if not return_stitches:
            return blended_image
        return blended_image, stitch1, stitch2, shift




    def create_stitch_images(self, ransac_params):
        stitch_image1, stitch_image2, shift = self.match_type.create_mosaic(self.align_solve_type, ransac_params)
        return stitch_image1, stitch_image2, shift
