import cv2
from PIL import Image
from Feature.ORBMatch import ORBMatch
import numpy as np
from Feature.AffineSolver import AffineSolver
import Stitching.ImageBlender as ImageBlender
import Stitching.ImageStitcher as ImageStitcher
from Stitching.MosaicMaker import MosaicMaker
import os
import timeit
from Toolbox.NamedArgs import NamedArgs
from Stitching.MultiMosaicer import MultiMosaicer
from Toolbox.ClassArgs import ClassArgs
from AlternateMethods.FullMatchMosaicer import FullMatchMosaicer
from Feature.HOGMatch import HOGMatch
from Feature.HOGMatchEdges import HOGMatchEdges
from Video.VideoToImageSaver import VideoToImageSaver
from Stitching.MosaicLoader import MosaicLoader
'''higher resolution seems to produce better mosaics (i.e. less likely for a frame to be lopsided in the mosaic)'''
RESIZE_DIMS = (426, 240)#(1280,720)#(4000, 3000)#(1280, 720)#(4000,3000)#(2400, 1800)#(4000, 3000)#(1600, 1200)




'''to do:

looks like mosaic'd image is too big (doesn't crop to bounds. Would be best to fix this mathematically instead of
just bounding it at each pass) (Image bounds extend too far to the right and downward)
Add a class that holds a class AND its params (for example an AlignSolve and its params, etc.)

Image blending only blends the image being transformed with the one image it is matched with,
causing rough edges at any other intersections

Scour code for temp images saved in arrays/etc and combine unneccesary uses of memory. Delete
temporary variables after use (using del <var>)


Rather than saving a full mosaic, save individual images along with a file that describes how to piece it together with the previous image

'''
video_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025.ts"
frame_save_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/"

'''
frame_save_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/"
vid_to_img = VideoToImageSaver(video_path, start_frame = 0)
vid_to_img.save_frames(frame_save_path)
'''

mosaic_image_base_path = frame_save_path#"E:/Big DZYNE Files/Mosaicing Data/Naplate East/"#"C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer/Mosaicing Data/"#
mosaic_save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/GEO Video Mosaics/Mosaics4/"
multi_mosaic_save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/GEO Video Mosaics/Mosaics5/"


start_time = timeit.default_timer()
multi_mosaicer = MultiMosaicer(multi_mosaic_save_path, mosaic_image_base_path, ".png", resize_dims = RESIZE_DIMS, start_index = 4319, num_images_to_mosaic = 50, index_skip = 8)
#all_mosaics = multi_mosaicer.create_mosaic(ORBMatch, AffineSolver, ClassArgs(ImageBlender.feather_blend, window_size = 15), NamedArgs(inlier_cutoff = 400, num_iter = 50))



'''CODE IN NEED OF MASSIVE CLEANUP'''
mosaic_loader = MosaicLoader(multi_mosaic_save_path, ".png")
loaded_mosaic_image = mosaic_loader.stitch_mosaic_at_index(1)
Image.fromarray(loaded_mosaic_image).show()

'''
for i in range(0, len(all_mosaics)):
    #cv2.imshow('Mosaic: ', all_mosaics[i])
    #cv2.waitKey(0)
    img = Image.fromarray(all_mosaics[i])
    img.show()
    img.save(mosaic_save_path + str(i) + ".png")
'''

'''
Image.fromarray(full_mosaic).save(mosaic_save_path + "Full-Mosaic-West-15-Images.png")
'''
print("time elapsed: ", timeit.default_timer() - start_time)
