import os
import cv2
import numpy as np

class Mosaic:

    IMAGE_PATH_EXTENSION = "Images/"
    MAT_PATH_EXTENSION = "Matrices/"
    '''Rather than working with one large mosaic image, it is less memory-taxing to
    save each image as is along with the transformation matrix that will fit
    it to the final, full mosaic image sequentially. This class handles saving the image
    and transformation matrix, as well as saving a README file describing how to piece the
    image back together again'''
    def __init__(self, save_path, image_extension = ".png"):
        self.save_path = save_path
        self.image_extension = image_extension
        self.current_path_index = 0
        self.current_save_index = 0
        self.update_current_paths()

    '''currently saves affine transformed and bounded images with a "shift" array
    that specifies how much the image has to be shifted to fit with the mosaic created
    up to that point'''
    def save_mosaic_info(self, mosaic_image, transformation_info):
        image_save_path = self.current_image_path + str(self.current_save_index) + self.image_extension
        transformation_info_save_path = self.current_mat_path + str(self.current_save_index)
        cv2.imwrite(image_save_path, mosaic_image)
        np.save(transformation_info_save_path, transformation_info)
        self.current_save_index += 1

    def tick_current_path_index(self):
        self.current_path_index += 1
        self.current_save_index = 0
        self.update_current_paths()


    def update_current_paths(self):
        self.current_mosaic_path = self.get_path_name_at_index(self.current_path_index)
        self.current_image_path = self.current_mosaic_path + Mosaic.IMAGE_PATH_EXTENSION
        self.current_mat_path = self.current_mosaic_path + Mosaic.MAT_PATH_EXTENSION
        if not os.path.exists(self.current_mosaic_path):
            os.makedirs(self.current_mosaic_path)
        if not os.path.exists(self.current_image_path):
            os.makedirs(self.current_image_path)
        if not os.path.exists(self.current_mat_path):
            os.makedirs(self.current_mat_path)

    def get_path_name_at_index(self, index):
        return self.save_path + str(index) + "/"
