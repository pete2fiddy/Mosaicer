import numpy as np
from PIL import Image

'''takes an image and returns a list of tuples where the first index is a numpy array representing the x and y coordinate of the color,
and the second index is the color'''
def image_to_points(image):
    points = []
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            points.append((np.array([x,y]), image[x,y]))
    return points


def get_image_corner_vectors(image):
    return [np.array([0,0]), np.array([0,image.shape[1]]), np.array([image.shape[0], image.shape[1]]), np.array([image.shape[0], 0])]

def norm_to_pil(image, clip_lower_bound = False):
    normed_image = image/float(np.amax(image)) if not clip_lower_bound else (image-np.amin(image))/(np.amax(image)-np.amin(image))
    return Image.fromarray(np.uint8(255*normed_image))

'''returns the bounding box of the image after it has been affine-transformed so that it will fit correctly in the window'''
'''def get_affine_transformed_image_bounds(affine_mat, image):
    dim = image.shape
    affine_image_corners = [np.array([0, 0, 1.0]), np.array([0, dim[1], 1.0]), np.array([dim[0], dim[1], 1.0]), np.array([dim[0], 0, 1.0])]
    transformed_image_corners = []
    for i in range(0, len(affine_image_corners)):
        append_transform = affine_mat.dot(affine_image_corners[i])[:2]
        transformed_image_corners.append(append_transform)
    '''
