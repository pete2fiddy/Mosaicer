import numpy as np
import ImageOp.ImageMath as ImageMath
import VectorOp.VectorMath as VectorMath
from PIL import Image
import cv2
import timeit

FEATURE_RECT_SIZE = 3


'''transforms image1 to fit image2, then returns two images, one for image1 and one for image2
that, when overlaid, would create a full image. The two are kept separate for blending purposes'''
def stitch_images(image1, image2, align_solve):
    solve_features = align_solve.solve_feature_matches
    #for i in range(0, len(solve_features)):
    #image1 = cv2.circle(image1, tuple(solve_features[i].xy1.astype(np.int)), 50, (255,0,0), thickness = 5)


    trans_image1, shift_image1 = align_solve.transform_image(image1)
    #trans_image1 = cv2.rectangle(trans_image1, (0,0), trans_image1.shape[:2][::-1], (0,0,255), thickness = 5)
    #Image.fromarray(trans_image1).show()
    #trans_origin1 = get_feature_track_delta(image1, image2, align_solve)
    #print("trans origin1 stitch1: ", trans_origin1)
    trans_origin1 = shift_image1#get_feature_track_delta4(shift_image1, trans_image1, image1, image2, align_solve)
    trans_origin1 = trans_origin1.astype(np.int)
    print("trans_origin1 is: ", trans_origin1)
    '''
    biggest_width = trans_image1.shape[0] if trans_image1.shape[0] > image2.shape[0] else image2.shape[0]
    biggest_height = trans_image1.shape[1] if trans_image1.shape[1] > image2.shape[1] else image2.shape[1]
    out_image_shape = (int(biggest_width + abs(trans_origin1[1])), int(biggest_height + abs(trans_origin1[0])), 3)
    '''

    out_image_shape = get_out_image_shape(trans_image1, image2, trans_origin1)

    print("out image shape: ", out_image_shape)

    out_image1 = np.zeros(out_image_shape)
    out_image2 = np.zeros(out_image_shape)


    xy1, xy2 = get_image_corners(trans_origin1)
    out_image1[xy1[0]:xy1[0] + trans_image1.shape[0], xy1[1]:xy1[1] + trans_image1.shape[1]] = trans_image1
    out_image2[xy2[0]:xy2[0] + image2.shape[0], xy2[1]:xy2[1] + image2.shape[1]] = image2

    #Image.fromarray(np.uint8(out_image1)).show()
    #Image.fromarray(np.uint8(out_image2)).show()

    return np.uint8(out_image1), np.uint8(out_image2), trans_origin1

'''stitches two images, the first of which is already transformed, using the "shift"
between the first and second image (used with saved data where transformed images are
saved as well as the shift to piece it together later, rather than saving one large file)'''
def stitch_with_shift(trans_image, base_image, shift):
    out_image1 = np.zeros((get_out_image_shape(trans_image, base_image, shift)))
    out_image2 = np.zeros((get_out_image_shape(trans_image, base_image, shift)))
    xy1, xy2 = get_image_corners(shift)
    out_image1[xy1[0]:xy1[0] + trans_image.shape[0], xy1[1]:xy1[1] + trans_image.shape[1]] = trans_image
    out_image2[xy2[0]:xy2[0] + base_image.shape[0], xy2[1]:xy2[1] + base_image.shape[1]] = base_image
    return np.uint8(out_image1), np.uint8(out_image2)


def get_out_image_shape(trans_image1, image2, trans_origin1):
    xy1, xy2 = get_image_corners(trans_origin1)

    trans_image1_startxy = (xy1[0], xy1[1])
    trans_image1_endxy = (trans_image1_startxy[0] + trans_image1.shape[0], trans_image1_startxy[1] + trans_image1.shape[1])

    image2_startxy = (xy2[0], xy2[1])
    image2_endxy = (image2_startxy[0] + image2.shape[0], image2_startxy[1] + image2.shape[1])

    leftest_x = 0
    rightest_x = trans_image1_endxy[0] if trans_image1_endxy[0] > image2_endxy[0] else image2_endxy[0]

    uppest_y = 0
    downest_y = trans_image1_endxy[1] if trans_image1_endxy[1] > image2_endxy[1] else image2_endxy[1]
    return (rightest_x - leftest_x, downest_y - uppest_y, 3)


'''
def get_feature_track_delta4(shift, trans_image1, image1, image2, align_solve):
    align_mat = align_solve.align_mat
    trans_origin_vec = np.array([align_mat[0,2], align_mat[1,2], 1])
    print("trans origin vec: ", trans_origin_vec)
    untrans_origin_vec = np.linalg.inv(align_mat).dot(trans_origin_vec)
    untrans_origin_vec = untrans_origin_vec[:2]

    solve_features = align_solve.solve_feature_matches
    xy1s_to_untrans_origin = []
    for i in range(0, len(solve_features)):
        xy1s_to_untrans_origin.append(solve_features[i].xy1 - untrans_origin_vec)

    trans_xy1s = align_solve.transform_points(xy1s_to_untrans_origin)
    #for i in range(0, len(trans_xy1s)):
        #cv2.circle(trans_image1, tuple((trans_xy1s[i] - trans_origin_vec[:2]).astype(np.int)), 50, (0,255,0), thickness = 5)

    return shift#np.array([0,0])
'''
def get_feature_track_delta3(trans_image1, image1, image2, align_solve):
    align_mat = align_solve.align_mat

    image1_corners = [np.array([0,0]), np.array([image1.shape[0],0]), np.array([image1.shape[0], image1.shape[1]]), np.array([0, image1.shape[1]])]
    trans_image1_corners = np.asarray(align_solve.transform_points(image1_corners)).astype(np.int)
    trans_bbox = cv2.boundingRect(trans_image1_corners)
    print("trans bbox: ", trans_bbox)
    print("trans corners: ", trans_image1_corners)
    print("trans corner [0][1]: ", trans_image1_corners[0][1])
    origin_add_y = (trans_bbox[2]+trans_bbox[0]) - trans_image1_corners[1][0]


    origin_translation = np.array([align_mat[0,2], align_mat[1,2]]).astype(np.int)
    print("origin translation: ", origin_translation)
    print("trans image1 dims: ", trans_image1.shape[:2])

    solve_features = align_solve.solve_feature_matches

    far_left_corner = np.array([0,0])
    for y in range(0, trans_image1.shape[0]):
        if trans_image1[y, 0][0] != 0:
            print("hit")
            far_left_corner = np.array([0,y])
            break

    print("far left corner: ", far_left_corner)
    origin_add_y = far_left_corner[1] - (trans_image1_corners[0] - origin_translation)[1]
    print("origin add Y: ", origin_add_y)
    cv2.circle(trans_image1, tuple((trans_image1_corners[0] - origin_translation).astype(np.int)), 80, (255,255,255), thickness = 5)
    #cv2.circle(trans_image1, tuple(far_left_corner), 20, (255,255,255), thickness = 2)
    trans_xy1s = []
    for i in range(0, len(solve_features)):
        trans_point = solve_features[i].xy1
        trans_xy1s.append(align_solve.transform_point(trans_point))
        #trans_xy1s[len(trans_xy1s)-1] -= origin_translation
    #trans_xy1s = align_solve.transform_feature_matches(solve_features)
    print("trans xy1s: ", trans_xy1s)
    circle_draw_points = []
    for i in range(0, len(trans_xy1s)):
        blue_point = np.array([trans_xy1s[i][0], trans_xy1s[i][1]])
        blue_point = blue_point - origin_translation
        blue_point[1] += origin_add_y
        #blue_point[1] += -origin_translation[1]
        #blue_point[1] -= 2*origin_translation[1]

        #trans_image1 = cv2.circle(trans_image1, tuple(blue_point.astype(np.int)), 20, (0,0,255), thickness = 2)
        #trans_image1 = cv2.circle(trans_image1, tuple((solve_features[i].xy2 - origin_translation).astype(np.int)), 25, (0,255,0), thickness = 2)
        circle_draw_points.append(tuple((trans_xy1s[i] - origin_translation).astype(np.int)))

    print("circle draw points: ", circle_draw_points)

    return np.array([origin_translation[0],  origin_translation[1] - origin_add_y])#origin_translation

'''only works with affine transform, as the midpoint of the bounding box of the transformed image only applies if the image is tarnsformed by affine. (I think)'''
def get_feature_track_delta2(trans_image1, image1, image2, align_solve):
    solve_features = align_solve.solve_feature_matches
    thresh_trans_image1 = cv2.cvtColor(trans_image1, cv2.COLOR_RGB2GRAY)
    thresh_trans_image1[thresh_trans_image1 > 0] = 255
    cv2.imshow("trans image 1: ", trans_image1)
    cv2.imshow("image2: ", image2)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    trans_corners = []

    start_time = timeit.default_timer()

    trans_corners.append(np.array([0,np.nonzero(thresh_trans_image1[0])[0][0]]))
    trans_corners.append(np.array([trans_image1.shape[0], np.nonzero(thresh_trans_image1[trans_image1.shape[0]-1])[0][0]]))

    trans_corners.append(np.array([np.nonzero(thresh_trans_image1[:, 0])[0][0], 0]))
    trans_corners.append(np.array([np.nonzero(thresh_trans_image1[:, thresh_trans_image1.shape[1]-1])[0][0], thresh_trans_image1.shape[1]-1]))

    trans_midpoint = np.average(trans_corners, axis = 0).astype(np.int)[::-1]
    print("trans midpoint: ", trans_midpoint)

    '''
    mask_contours = cv2.findContours(thresh_trans_image1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    mask_contours_sorted_by_size = sorted(mask_contours, key = lambda contour: len(contour), reverse = True)
    biggest_contour = mask_contours_sorted_by_size[0]
    biggest_contour = biggest_contour[:, 0, :]
    print("biggest contour: ", biggest_contour)
    trans_midpoint = np.average(biggest_contour, axis = 0).astype(np.int)[::-1]
    print("trans midpoint: ", trans_midpoint)'''



    image1_midpoint = np.asarray(image1.shape[:2])[::-1]/2.0


    trans_image1_midpoint = align_solve.transform_point(image1_midpoint)
    trans_image1_midpoint_to_trans_image_delta = trans_midpoint - trans_image1_midpoint

    feature_xy1s_on_trans_image1 = []
    for i in range(0, len(solve_features)):
        trans_solve_feature_xy1 = align_solve.transform_point(solve_features[i].xy1)[::-1] + trans_image1_midpoint_to_trans_image_delta
        feature_xy1s_on_trans_image1.append(trans_solve_feature_xy1)
        print("diff between: ", trans_solve_feature_xy1[::-1] - solve_features[i].xy2)

    #Image.fromarray(test_image).show()



    feature_deltas = []
    for i in range(0, len(feature_xy1s_on_trans_image1)):
        feature_deltas.append(feature_xy1s_on_trans_image1[i] - solve_features[i].xy2[::-1])

    avg_feature_delta = np.average(feature_deltas, axis = 0)
    avg_feature_delta = avg_feature_delta[::-1]
    #avg_feature_delta[0] = -avg_feature_delta[0]

    return np.array([0,0])#-avg_feature_delta.astype(np.int)





'''
Function created to fix the following:

When an align solve object returns the transformed image, it returns an image fit to the bounds of the edges of the transformed image.
You can't just plop that image on top of the first, and the translation required to transform that image to fit the base is lost when CV affineTransforms the image.
What this method does is PHYSICALLY track where the features move to during the affine transform with respect to the image, rather than with respect to the vector
transformation. This allows you to measure how much the transformed image needs to move to fit the same features present in the original image
'''
def get_feature_track_delta(image1, image2, align_solve):
    feature_image1 = np.zeros((image1.shape[:2]))
    feature_image2 = np.zeros((image2.shape[:2]))

    solve_features = align_solve.solve_feature_matches
    trans_solve_feature_images1 = []
    trans_feature_xys = []
    for i in range(0, len(solve_features)):
        trans_feature_image = np.zeros((image1.shape[:2]))
        feat_xy1 = solve_features[i].xy1.astype(np.int)
        '''feature must be drawn with an (unneccesarily) large box instead of as a single
        or small clump of pixels. Depending on the affine transform, the image can be aggressively
        transformed such that the entire "white" feature is removed completely given that the majority
        of the image is black. The average, which is taken at the end, is still the same. (not the case
        for features that clip the edge of the image)

        To add:
        Size of margin should be adjusted to the size of the picture

        This is still a hack of a fix. Would be best to just transform the vectors
        of the features, but was having many issues with coping with how images must line up afterward'''
        trans_feature_image[feat_xy1[1] - FEATURE_RECT_SIZE //2:feat_xy1[1]+ (FEATURE_RECT_SIZE//2)+1, feat_xy1[0]-FEATURE_RECT_SIZE//2:feat_xy1[0]+(FEATURE_RECT_SIZE//2)+1] = 255

        trans_feature_image = align_solve.transform_image(trans_feature_image)

        ith_feature_xs, ith_feature_ys = np.where(trans_feature_image > 0)

        trans_feature_xys.append(np.array([np.average(ith_feature_ys), np.average(ith_feature_xs)]))

    feature_deltas = []


    for i in range(0, len(trans_feature_xys)):
        append_delta = solve_features[i].xy2 - trans_feature_xys[i]
        feature_deltas.append(append_delta)
    '''error in taking average sometimes found here, where "average of an empty slice" is thrown.
    When this is thrown, the mosaic is incorrect, and causes (for the most part) all subsequent
    mosaics to also be incorrect. Seems to be a result of bad feature matching (tends to occur when the
    number of inlier feature pairs RANSAC finds is < 10)
    Given that the data is GPS Stamped, may be best to fall back on GPS transforms for images
    that do not conform correctly'''

    avg_trans_delta = np.average(feature_deltas, axis = 0).astype(np.int)
    return avg_trans_delta


def get_image_corners(trans_origin1):
    '''if [1] is positive, x1 = [1]
    if [1] is negative, x2 = abs([1])'''
    x1 = 0
    x2 = 0
    if trans_origin1[1] > 0:
        x1 = abs(trans_origin1[1])
    else:
        x2 = abs(trans_origin1[1])


    '''if [0] is positive, y1 = [0]
    if [0] is negative, y2 = abs([0])'''
    y1 = 0
    y2 = 0
    if trans_origin1[0] > 0:
        y1 = abs(trans_origin1[0])
    else:
        y2 = abs(trans_origin1[0])

    return (x1,y1), (x2, y2)


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
