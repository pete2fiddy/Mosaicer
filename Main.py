import cv2
from PIL import Image
from Feature.ORBMatch import ORBMatch

RESIZE_DIMS = (450, 600)
mosaic_image_base_path = "/Users/phusisian/Desktop/DZYNE/Mosaicing Data/"
image1 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0145.JPG"), RESIZE_DIMS)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(cv2.imread(mosaic_image_base_path + "DJI_0146.JPG"), RESIZE_DIMS)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

orb_match = ORBMatch(image1, image2)
feature_image1, feature_image2 = orb_match.draw_features()

#Image.fromarray(feature_image1).show()
Image.fromarray(feature_image2).show()
