from Feature.MatchType import MatchType
import numpy as np
from Feature.FeatureMatch import FeatureMatch
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from Feature.AffineSolver import AffineSolver
from Feature.RANSAC import RANSAC


class ORBMatch(MatchType):

    def __init__(self, image1, image2):
        MatchType.__init__(self, image1, image2)


    def init_features(self):
        orb = cv2.ORB_create()
        image1_keypoints, self.image1_descriptors = orb.detectAndCompute(self.image1, None)
        image2_keypoints, self.image2_descriptors = orb.detectAndCompute(self.image2, None)
        self.set_features(image1_keypoints, image2_keypoints)

    def match_features(self):
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        kp_matches = brute_force_matcher.match(self.image1_descriptors, self.image2_descriptors)
        feature_matches = FeatureMatch.cv_matches_to_feature_matches(kp_matches, self.features1, self.features2)

        #affine_solver = AffineSolver(feature_matches[9:12], feature_matches)
        ransac = RANSAC(feature_matches, AffineSolver, 1500)
        affine_solver = ransac.fit()
        transformed_by_affine_image = affine_solver.transform_image(self.image1)
        Image.fromarray(transformed_by_affine_image).show()
