from abc import ABC, abstractmethod
import Feature.CVConverter as CVConverter
import cv2
from Feature.RANSAC import RANSAC
import Stitching.ImageStitcher as ImageStitcher

'''MatchType is an abstract class that gives and defines all functionalities that a given two-image feature-matching
algorithm might required for the remainder of the code'''
class MatchType(ABC):
    '''image1 and image2 must be numpy images'''
    def __init__(self, image1, image2, mask = None):
        self.image1 = image1
        self.image2 = image2
        self.mask = mask
        self.feature_matches = None
        self.init_features()
        self.match_features()

    '''in the future replace ransac_params with the "pass arguments as tuple" thing I learned from Nicholas'''
    def create_mosaic(self, align_solve, ransac_params):
        align_solver = self.create_align_solver(align_solve, ransac_params)
        return ImageStitcher.stitch_images(self.image1, self.image2, align_solver)

    def create_align_solver(self, align_solve_type, ransac_params):
        ransac = RANSAC(self.feature_matches, align_solve_type)
        ransac.set_params(ransac_params)
        align_solver = ransac.fit()
        return align_solver


    '''through whatever means the match type allows, the required "init_features" method finds the salient points in both images
    (with no regards to feature matching yet)'''
    @abstractmethod
    def init_features(self):
        pass

    '''through whatever means the match type allows, the required "match_features" method matches a salient feature in the first image
    to its location in the second image.

    These matches must be saved in the correct format by calling the setter method "set_matches". They can be received using the getter
    method "get_matches"'''
    @abstractmethod
    def match_features(self):
        pass

    '''returns a copy of image1 and image2 where the features are drawn to both with the specified (or default) color'''
    def draw_features(self, radius = 5, thickness = 1, color = (255,0,0)):
        image1_out = self.image1.copy()
        image2_out = self.image2.copy()
        for i in range(0, len(self.features1)):
            cv2.circle(image1_out, CVConverter.kp_to_tuple_int(self.features1[i])[0:2], radius, color, thickness = thickness)
        for i in range(0, len(self.features2)):
            cv2.circle(image2_out, CVConverter.kp_to_tuple_int(self.features2[i])[0:2], radius, color, thickness = thickness)
        return image1_out, image2_out


    '''features1 are the point positions of the features in image1, and features2 are the point positions of the features in image2'''
    def set_features(self, features1, features2):
        self.features1 = features1
        self.features2 = features2

    def set_matches(self, matches):
        self.feature_matches = matches

    def get_matches(self):
        return self.feature_matches
