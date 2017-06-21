import numpy as np
import Feature.CVConverter as CVConverter

'''the FeatureMatch class is used to hold a reference to an XY position in "image1" that contains a feature that appears in an XY position
in "image2"'''
class FeatureMatch:

    '''
    Numpy images become reversed by 90 degrees. As such, the xy positions listed *will* be in xy form, but would be xy of the image
    after it is rotated 90 degrees. It is easiest to just keep it this way as it will unify opencv operations and all math will just be
    "unflipped" once the transformation is unapplied in converting it back to a correctly oriented image

    For sake of simplicity, should do little other than hold an xy1 and xy2 and perhaps have some scoring methods to determine
    the quality of fit of A.xy1 (approximately =) xy2

    Chose to do this because it is often problematic thinking about switching between two formats: many opencv and numpy
    methods rely on xy being yx -- keeping it consistent throughout should lead to no issues, despite it being unintuitive. (all
    transformations will just be flipped, which will lead to the same end result since the images are flipped anyway)'''
    def __init__(self, xy1, xy2):
        self.xy1 = xy1
        self.xy2 = xy2


    '''converts opencv's Matcher objects to FeatureMatch objects using the match objects, the xy positions of features in image1,
    and the xy positions of features in image2'''
    @staticmethod
    def cv_matches_to_feature_matches(cv_matches, features1, features2):
        out_feature_matches = []
        '''query image is image1, train image is image2'''
        for i in range(0, len(cv_matches)):
            feature1_index = cv_matches[i].queryIdx
            feature2_index = cv_matches[i].trainIdx
            append_feature_match = FeatureMatch(CVConverter.kp_to_np(features1[feature1_index])[:2], CVConverter.kp_to_np(features2[feature2_index])[:2])
            out_feature_matches.append(append_feature_match)
        return out_feature_matches

    def delta(self):
        return self.xy2 - self.xy1

    def transform(self, align_solve):
        p1_transformed = align_solve.transform_feature_match(self)
        return FeatureMatch(p1_transformed, self.xy2)

    def dist_to_xy2(self, compare_point):
        return np.linalg.norm(compare_point - self.xy2)

    def __repr__(self):
        return "<{0} | {1}>".format(self.xy1, self.xy2)
