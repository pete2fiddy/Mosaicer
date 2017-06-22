import numpy as np
import ImageOp.ImageMath as ImageMath
import VectorOp.VectorMath as VectorMath
from PIL import Image
import cv2



'''transforms image1 to fit image2, then returns two images, one for image1 and one for image2
that, when overlaid, would create a full image. The two are kept separate for blending purposes'''
def stitch_images_old(image1, image2, align_solve):
    trans_image1, trans_origin1 = align_solve.transform_image(image1)
    trans_origin1 = trans_origin1.astype(np.int)

    biggest_width = trans_image1.shape[0] if trans_image1.shape[0] > image2.shape[0] else image2.shape[0]
    biggest_height = trans_image1.shape[1] if trans_image1.shape[1] > image2.shape[1] else image2.shape[1]

    out_image_shape = (int(biggest_width + abs(trans_origin1[1])), int(biggest_height + abs(trans_origin1[0])), 3)
    print("out image shape: ", out_image_shape)

    out_image1 = np.zeros(out_image_shape)
    out_image2 = np.zeros(out_image_shape)

    x1 = trans_origin1[1]
    '''crashes when x1 is < 0'''
    print("x1: ", x1)
    x2 = 0
    y1 = 0
    y2 = abs(trans_origin1[0])
    out_image1[x1:x1+trans_image1.shape[0], y1 : y1+trans_image1.shape[1]] = trans_image1
    out_image2[x2:x2+image2.shape[0], y2:y2+image2.shape[1]] = image2
    return np.uint8(out_image1), np.uint8(out_image2)


'''transforms image1 to fit image2, then returns two images, one for image1 and one for image2
that, when overlaid, would create a full image. The two are kept separate for blending purposes'''
def stitch_images(image1, image2, align_solve):
    align_solve.draw_features(image1, image2, 20, thickness = 5)

    trans_image1 = align_solve.transform_image(image1)
    trans_origin1 = get_feature_track_delta(image1, image2, align_solve)
    #trans_origin1 = np.array([0,0])
    trans_origin1 = trans_origin1.astype(np.int)
    '''the image at image1 must be moved by:
    x: trans_origin1[0]
    y: trans_origin1[1]
    given that they have the same top left corner to begin with in order for the transformation to match'''

    biggest_width = trans_image1.shape[0] if trans_image1.shape[0] > image2.shape[0] else image2.shape[0]
    biggest_height = trans_image1.shape[1] if trans_image1.shape[1] > image2.shape[1] else image2.shape[1]

    out_image_shape = (int(biggest_width + abs(trans_origin1[1])), int(biggest_height + abs(trans_origin1[0])), 3)
    print("out image shape: ", out_image_shape)

    out_image1 = np.zeros(out_image_shape)
    out_image2 = np.zeros(out_image_shape)


    xy1, xy2 = get_image_corners(trans_origin1)
    out_image1[xy1[0]:xy1[0] + trans_image1.shape[0], xy1[1]:xy1[1] + trans_image1.shape[1]] = trans_image1
    out_image2[xy2[0]:xy2[0] + image2.shape[0], xy2[1]:xy2[1] + image2.shape[1]] = image2
    return np.uint8(out_image1), np.uint8(out_image2)


def get_feature_track_delta(image1, image2, align_solve):
    feature_image1 = np.zeros((image1.shape[:2]))
    feature_image2 = np.zeros((image2.shape[:2]))

    solve_features = align_solve.solve_feature_matches
    for i in range(0, 1):
        feat_xy1 = solve_features[i].xy1.astype(np.int)
        feat_xy2 = solve_features[i].xy2.astype(np.int)
        feature_image1[feat_xy1[1], feat_xy1[0]] = i+1
        feature_image2[feat_xy2[1], feat_xy2[0]] = i+1

    trans_feature_image1 = align_solve.transform_image(feature_image1)

    trans_feature_indexes = np.nonzero(trans_feature_image1)
    print("trans feature indexes: ", trans_feature_indexes)
    trans_feature_xys = []
    for i in range(0, len(trans_feature_indexes)):
        trans_feature_xys.append(np.array([trans_feature_indexes[1][i], trans_feature_indexes[0][i]]))

    new_feature_xy = np.average(trans_feature_xys, axis = 0).astype(np.int)
    trans_image1 = align_solve.transform_image(image1)
    trans_feature_image1 = cv2.circle(trans_image1, tuple(new_feature_xy), 30, (0,255,0), thickness = 5)
    Image.fromarray(trans_feature_image1).show()

    print('trans feature xys: ', trans_feature_xys)
    print("new feature xy: ", new_feature_xy)
    new_trans_origin = (solve_features[0].xy2 - new_feature_xy).astype(np.int)
    print('new trans origin: ', new_trans_origin)
    return new_trans_origin


def get_image_corners(trans_origin1):
    print("trans origin: ", trans_origin1)
    '''if [1] is positive, x1 = [1]
    if [1] is negative, x2 = abs([1])'''
    x1 = 0
    x2 = 0
    if trans_origin1[1] > 0:
        print("x1 if hit")
        '''worked fine: 1'''
        x1 = abs(trans_origin1[1])
    else:
        '''worked fine: 1'''
        print("x2 if hit")
        x2 = abs(trans_origin1[1])


    '''if [0] is positive, y1 = [0]
    if [0] is negative, y2 = abs([0])'''
    y1 = 0
    y2 = 0
    if trans_origin1[0] > 0:
        print("y1 if hit")
        '''worked fine : 0,
        didn't work: 1'''
        y1 = abs(trans_origin1[0])
        #y2 = 400
    else:
        '''worked fine: 1'''
        print("y2 if hit")
        y2 = abs(trans_origin1[0])

    print("shift output: ", (x1, y1), (x2, y2))

    if trans_origin1[0] < 0 and trans_origin1[1] < 0:
        print('if1 hit')
    elif trans_origin1[0] < 0 and trans_origin1[1] > 0:
        print('if2 hit')
        '''appears to be working properly: 4'''
    elif trans_origin1[0] > 0 and trans_origin1[1] < 0:
        print("if3 hit")
        '''does not appear to be working properly: 1
        does appear to be working properly: 3'''
    elif trans_origin1[0] > 0 and trans_origin1[1] > 0:
        print('if4 hit')

    #return (0,0), (73,311)

    return (x1,y1), (x2, y2)
    #return(0,0),(0,0)


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
