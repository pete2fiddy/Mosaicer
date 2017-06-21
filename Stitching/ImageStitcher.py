import numpy as np
import ImageOp.ImageMath as ImageMath
import VectorOp.VectorMath as VectorMath
from PIL import Image
import cv2



'''transforms image1 to fit image2, then returns two images, one for image1 and one for image2
that, when overlaid, would create a full image. The two are kept separate for blending purposes'''
def stitch_images(image1, image2, align_solve):
    trans_image1, trans_origin1 = align_solve.transform_image(image1)
    trans_origin1 = trans_origin1.astype(np.int)

    biggest_width = trans_image1.shape[0] if trans_image1.shape[0] > image2.shape[0] else image2.shape[0]
    biggest_height = trans_image1.shape[1] if trans_image1.shape[1] > image2.shape[1] else image2.shape[1]

    out_image_shape = (int(biggest_width + abs(trans_origin1[1])), int(biggest_height + abs(trans_origin1[0])), 3)
    out_image1 = np.zeros(out_image_shape)
    out_image2 = np.zeros(out_image_shape)

    x1 = trans_origin1[1]
    print("x1: ", x1)
    x2 = 0
    y1 = 0
    y2 = abs(trans_origin1[0])
    out_image1[x1:x1+trans_image1.shape[0], y1 : y1+trans_image1.shape[1]] = trans_image1
    out_image2[x2:x2+image2.shape[0], y2:y2+image2.shape[1]] = image2
    return np.uint8(out_image1), np.uint8(out_image2)

'''creates a mask that is white (255) wherever the two images are present'''
def get_mask(stitch_image1, stitch_image2):
    gray_image1= cv2.cvtColor(stitch_image1, cv2.COLOR_RGB2GRAY)
    gray_image2 = cv2.cvtColor(stitch_image2, cv2.COLOR_RGB2GRAY)
    bool_image1 = np.zeros(stitch_image1.shape[:2],dtype = np.bool)
    bool_image2 = np.zeros(stitch_image2.shape[:2], dtype = np.bool)
    bool_image1[gray_image1 > 0] = True
    bool_image2[gray_image2 > 0] = True
    out_image = np.uint8(255*np.logical_or(bool_image1, bool_image2).astype(np.int))
    return out_image

def get_single_stitch_mask(stitch_image):
    thresh_image= cv2.cvtColor(stitch_image, cv2.COLOR_RGB2GRAY)
    thresh_image[thresh_image > 0 ]= 255.0
    return thresh_image
