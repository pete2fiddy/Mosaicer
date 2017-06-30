import Stitching.ImageStitcher as ImageStitcher
from Stitching.Mosaic import Mosaic
import os
import cv2
import numpy as np
import Stitching.ImageBlender as ImageBlender
from PIL import Image

class MosaicLoader:
    '''takes the path to mosaics saved using a Mosaic object and pieces the
    images together into one large mosaic'''
    def __init__(self, load_path, image_extension):
        self.load_path = load_path
        self.image_extension = image_extension

    def stitch_mosaic_at_index(self, index):
        path_at_index = self.load_path + str(index) + "/"
        image_path_at_index = path_at_index + Mosaic.IMAGE_PATH_EXTENSION
        mat_path_at_index = path_at_index + Mosaic.MAT_PATH_EXTENSION

        image_paths = self.get_paths_in_path_with_extension(image_path_at_index, self.image_extension)
        mat_paths = self.get_paths_in_path_with_extension(mat_path_at_index, ".npy")

        first_image, first_mat = self.load_image_and_mat_at_index(image_paths[0], mat_paths[0])
        full_mosaic = first_image
        for i in range(1, len(image_paths)):
            append_image, append_mat = self.load_image_and_mat_at_index(image_paths[i], mat_paths[i])
            stitch1, stitch2 = ImageStitcher.stitch_with_shift(append_image, full_mosaic, append_mat)

            full_mosaic = ImageBlender.paste_blend(stitch1, stitch2)
            Image.fromarray(full_mosaic).show()
        return full_mosaic

    def load_image_and_mat_at_index(self, images_path, mat_path):
        image = cv2.imread(images_path)
        mat = np.load(mat_path)
        return image, mat

    def get_paths_in_path_with_extension(self, base_path, extension):
        names = os.listdir(base_path)
        names.sort(key = lambda name: int(name[:name.find(extension)]))
        paths = []
        for i in range(0, len(names)):
            if extension in names[i]:
                paths.append(base_path + names[i])
        return paths
