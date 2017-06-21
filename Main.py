import cv2
from PIL import Image
from Feature.ORBMatch import ORBMatch
import numpy as np
from Feature.AffineSolver import AffineSolver
import Stitching.ImageBlender as ImageBlender
import Stitching.ImageStitcher as ImageStitcher
import os

RESIZE_DIMS = (4000, 3000)#(1800, 2400)#(900, 1200)#(450, 600)



mosaic_image_base_path = "/Users/phusisian/Desktop/DZYNE/Mosaicing Data/"
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



image1 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0146.JPG"), RESIZE_DIMS)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0147.JPG"), RESIZE_DIMS)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

orb_match = ORBMatch(image1, image2, None)
stitch_image1, stitch_image2 = orb_match.create_mosaic(AffineSolver, 5000)
Image.fromarray(ImageBlender.feather_blend(stitch_image1, stitch_image2, 81)).show()
Image.fromarray(ImageStitcher.get_mask(stitch_image1, stitch_image2)).show()
feature_image1, feature_image2 = orb_match.draw_features()
