from abc import ABC, abstractmethod
import Feature.CVConverter as CVConverter
from Feature.FeatureMatch import FeatureMatch
import cv2

'''AlignSolver is an abstract class for different type of image transformations. The input is a list of feature matches
from two images and the class solves for the transformation matrix that allows one image to match with the other correctly.'''
class AlignSolve(ABC):
    '''takes a "solve" list of FeatureMatch objects. These (subset of points) are used to solve for the transformation matrix.
    Also takes an "all_feature_matches" list of FeatureMatch objects. These have no effect on calculating the transformation
    matrix, but are used to judge the cost function of the transformation'''
    def __init__(self, solve_feature_matches, all_feature_matches):
        self.solve_feature_matches = solve_feature_matches
        self.all_feature_matches = all_feature_matches
        self.align_mat = None
        self.solve_mat()

    '''the transformation matrix to allow image1 to transform to fit image2 is calculated and saved as "align_mat"'''
    @abstractmethod
    def solve_mat(self):
        pass

    '''Transforms inputted image so that it aligns with image2 using the solved for "align_mat"'''
    @abstractmethod
    def transform_image(self, image):
        pass


    '''Transforms the inputted feature1 in a FeatureMatch object. (Different solvers have different ways of representing vectors so
    it will not be universal for each solution).
    Returns the FeatureMatch object's xy1 transformed with the alignment matrix'''
    @abstractmethod
    def transform_feature_match(self, feature_match):
        pass

    def transform_feature_matches(self, feature_matches):
        out_matches = []
        for i in range(0, len(feature_matches)):
            out_matches.append(self.transform_feature_match(feature_matches[i]))
        return out_matches

    '''is a little bit clunky but leverages the fact that all AlignSolve subclasses have to have a feature match transformation object
    so you don't have to recode each time'''
    def transform_point(self, point):
        point_feature_match = FeatureMatch(point, point)
        out_point = self.transform_feature_match(point_feature_match)
        return out_point

    def transform_points(self, points):
        transformed_points = []
        for i in range(0, len(points)):
            transformed_points.append(self.transform_point(points[i]))
        return transformed_points

    def transform_image_points(self, image_points):
        transformed_image_points = []
        for i in range(0, len(image_points)):
            xy = image_points[i][0]
            transformed_xy = self.transform_point(xy)
            append_image_point = (transformed_xy, image_points[i][1])
            transformed_image_points.append(append_image_point)
        return transformed_image_points

    '''returns as a list of tuples each FeatureMatch object where xy1 is transformed by the alignment matrix and xy2 is left unchanged'''
    def get_transformed_match_pairs(self):
        match_pairs = []
        for i in range(0, len(self.all_feature_matches)):
            transformed_xy1 = self.transform_feature_match(self.all_feature_matches[i])
            append_pair = (transformed_xy1, self.all_feature_matches[i].xy2)
            match_pairs.append(append_pair)
        return match_pairs

    def draw_features(self, image1, image2, radius, thickness = 5):
        for i in range(0, len(self.solve_feature_matches)):
            image1, image2 = self.solve_feature_matches[i].draw(image1, image2, radius, thickness = thickness)

    '''used to ensure that the image that contains the transformed image is clipped so that it perfectly bounds the transformed image (sometimes is not the case when
    transformed image bounding box is precalculated)'''
    @staticmethod
    def get_transformed_image_bounds(trans_image):
        thresh_trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2GRAY)
        thresh_trans_image[thresh_trans_image > 0] = 255
        thresh_contour = cv2.findContours(thresh_trans_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour_bbox = cv2.boundingRect(thresh_contour)
        return contour_bbox

    @staticmethod
    def crop_transformed_image_to_bounds(trans_image):
        bounds = AlignSolve.get_transformed_image_bounds(trans_image)
        return trans_image[bounds[1] : bounds[3], bounds[0] : bounds[2]]

    '''holds the number of required points for the alignment object to solve for a transformation matrix'''
    @staticmethod
    @abstractmethod
    def NUM_SOLVE_FEATURES():
        pass
