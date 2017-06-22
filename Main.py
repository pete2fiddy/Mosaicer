import cv2
from PIL import Image
from Feature.ORBMatch import ORBMatch
import numpy as np
from Feature.AffineSolver import AffineSolver
import Stitching.ImageBlender as ImageBlender
import Stitching.ImageStitcher as ImageStitcher
from Stitching.MosaicMaker import MosaicMaker
import os
from Toolbox.NamedArgs import NamedArgs

RESIZE_DIMS = (2400, 1800)#(900, 1200)#(450, 600)## (4000, 3000)#



mosaic_image_base_path = "C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer/Mosaicing Data/"
images = []
'''

image1 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0143.JPG"), RESIZE_DIMS)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0144.JPG"), RESIZE_DIMS)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

orb_match = ORBMatch(image1, image2, None)
stitch_image1, stitch_image2 = orb_match.create_mosaic(AffineSolver, 1500)
Image.fromarray(ImageBlender.feather_blend(stitch_image1, stitch_image2, 21)).show()
ImageStitcher.get_mask(stitch_image1, stitch_image2)
'''
'''
image_names = os.listdir(mosaic_image_base_path)
print("image names: ", image_names)
image_names.sort()
for i in range(0, len(image_names)):
    image_name = image_names[i]
    if ".JPG" in image_name:
        image_path = mosaic_image_base_path + image_name
        append_image = cv2.resize(cv2.imread(image_path), RESIZE_DIMS)
        append_image = cv2.cvtColor(append_image, cv2.COLOR_BGR2RGB)
        images.append(append_image)
'''
'''
start_index = 1
end_index = 3
mask = None
full_image = None
for i in range(start_index, end_index):
    image1 = images[i+1]
    image2 = full_image
    if i == start_index:
        image2 = images[i]

    orb_match = ORBMatch(image1, image2, mask)
    stitch_image1, stitch_image2 = orb_match.create_mosaic(AffineSolver, 1500)
    Image.fromarray(stitch_image1).show()
    Image.fromarray(stitch_image2).show()
    full_image = ImageBlender.feather_blend(stitch_image1, stitch_image2, 21)
    Image.fromarray(full_image).show()
    mask = ImageStitcher.get_mask(stitch_image1, stitch_image2)
    Image.fromarray(mask).show()

Image.fromarray(full_image).show()
Image.fromarray(mask).show()
'''



image1 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0148.JPG"), RESIZE_DIMS)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image2 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0147.JPG"), RESIZE_DIMS)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#image2 = image2[:RESIZE_DIMS[0]-500, :RESIZE_DIMS[1]-500]

mosaic_maker = MosaicMaker(image1, image2, ORBMatch, AffineSolver, ImageBlender.paste_blend)
mosaic_image = mosaic_maker.create_mosaic(NamedArgs(num_iter = 1000), NamedArgs(window_size = 81))
Image.fromarray(mosaic_image).show()
#mosaic_mask = ImageStitcher.get_single_stitch_mask(mosaic_image)
#Image.fromarray(mosaic_mask).show()
