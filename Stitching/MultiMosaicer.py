import os
import cv2
from Stitching.MosaicMaker import MosaicMaker
import Stitching.ImageStitcher as ImageStitcher
from PIL import Image
import numpy as np
from Stitching.Mosaic import Mosaic
import ImageOp.ImageMath as ImageMath
from Feature.AlignSolve import AlignSolve

class MultiMosaicer:
    '''couldn't think of a better name for the below. When an error is found in creating a mosaic, it is assumed that some number of previous frames
    are also erroneous. When an error occurs in creating a mosaic, the current mosaic is saved and a new one is started using the image that caused an error.
    However, often these errors are a result of past mistakes in the mosaicing process, so this number quantifies how many images to remove from the previous
    mosaic before saving, and how many indexes to move back to arrive at the index of the image that the mosaic is restarted from'''
    MOSAIC_ERROR_RESTART_INDEX = 0
    '''

    image_path: the path to the images, must end with "/". Assumes all images in the path are sorted
    numerically/alphabetically so that python's "sort" method can order them correctly
    image_extension: the extension type of the image (e.g. ".JPG")'''
    def __init__(self, save_path, image_path, image_extension, resize_dims = None, start_index = None, num_images_to_mosaic = None, index_skip = None):
        self.image_path = image_path
        self.image_extension = image_extension
        self.resize_dims = resize_dims
        self.init_image_paths()
        self.index_skip = index_skip if index_skip is not None else 1
        temp_start_index = start_index if start_index != None else 0
        temp_end_index = len(self.image_paths) if num_images_to_mosaic == None else temp_start_index + num_images_to_mosaic * self.index_skip
        self.image_paths = self.image_paths[temp_start_index : temp_end_index : self.index_skip]
        #self.num_images_to_mosaic = num_images_to_mosaic if num_images_to_mosaic != None else len(self.image_paths)
        self.mosaic = Mosaic(save_path)
        print("image paths: ", self.image_paths)


    def init_image_paths(self):
        self.image_paths = os.listdir(self.image_path)

        print("index of .png in first name: ", self.image_paths[0][:self.image_paths[0].find(self.image_extension)])
        self.image_paths.sort(key = lambda name: int(name[:name.find(self.image_extension)]))

        i = 0
        while i < len(self.image_paths):
            if self.image_extension not in self.image_paths[i]:
                del self.image_paths[i]
            else:
                image_path = self.image_path + self.image_paths[i]
                self.image_paths[i] = image_path
                i+=1


    def load_image_by_index(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.resize_dims is not None:
            image = cv2.resize(image, self.resize_dims)
        return image

    '''image1 passed to the align solver should NOT be a mask of the entire last mosaic,
    should instead be the previous image in its correct position in the full mosaic, but with no
    other images in the mosaic present. This prevents bad seams from tampering with the results of
    future images'''
    '''should add a class that can hold both a class and its params so that I can chop the number
    of arguments this takes in half'''
    '''feature_match is the class of feature map type used to mosaic
    align_solve is the class of align solve type used to mosaic
    blend type is the method of blending used'''
    '''would be best if this could hold all the affine transformed images
    separately until the end where it translates and merges them all together.
    This would preserve the feathered edges of the images, and saves memory? (may not work)'''
    def create_mosaic(self, feature_match, align_solve, blend_func_and_args, ransac_params):
        last_mosaic = self.load_image_by_index(0)
        last_mask = None

        all_mosaics = []
        i = 1
        while i < len(self.image_paths):
            image1 = self.load_image_by_index(i)
            new_mosaic = None
            '''issue with the below is that, while it DOES catch bad mosaics, most bad mosaics (that would cause a crash otherwise) are a result of the previous
            image being lopsidedly mosaiced. Would be best to:
            1) remove the previous mosaic from the last mosaic somehow
            2) Start from the previous image as the identity, not the one at i (start at i-1 instead)

            Also, to prevent "bad mosaic transforms," may be possible to add some constraints that can be determined if a transformed image is unreasonably squashed

            '''
            try:
                mosaic_maker = MosaicMaker(image1, last_mosaic, feature_match, align_solve, mask = last_mask)
                new_mosaic, stitch1, stitch2, shift = mosaic_maker.create_mosaic(ransac_params, blend_func_and_args, return_stitches = True)
                last_mask = ImageStitcher.get_single_stitch_mask(stitch1)
                self.mosaic.save_mosaic_info(AlignSolve.crop_transformed_image_to_bounds(stitch1), shift)
            except:
                self.mosaic.tick_current_path_index()
                print('=======================')
                print("new mosaic broken off")
                print('=======================')
                Image.fromarray(last_mosaic).show()
                all_mosaics.append(last_mosaic.copy())
                if i > MultiMosaicer.MOSAIC_ERROR_RESTART_INDEX:
                    new_mosaic = self.load_image_by_index(i-MultiMosaicer.MOSAIC_ERROR_RESTART_INDEX)
                    i -= MultiMosaicer.MOSAIC_ERROR_RESTART_INDEX
                else:
                    new_mosaic = self.load_image_by_index(i)
                last_mask = None
            
            last_mosaic = new_mosaic
            i+=1



        all_mosaics.append(last_mosaic)
        return all_mosaics
