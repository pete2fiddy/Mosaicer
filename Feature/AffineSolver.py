from Feature.AlignSolve import AlignSolve
import numpy as np
import VectorOp.VectorMath as VectorMath
import cv2
from PIL import Image

class AffineSolver(AlignSolve):
    '''solves for mosaicing using only affine transforms.
    Affine matrixes are ones whose bottom row is restricted to [0,0,1].
    By using a 3x3 matrix where the bottom row is [0,0,1], some non-linear transformations such as
    translation are possible'''
    def __init__(self, solve_feature_matches, all_feature_matches):
        if len(solve_feature_matches) != AffineSolver.NUM_SOLVE_FEATURES():
            print("ERROR: AFFINE SOLVER GIVEN A SET OF MATCHES WITH LENGTH GREATER THAN 3")
        AlignSolve.__init__(self, solve_feature_matches, all_feature_matches)


    def solve_mat(self):
        '''saves the transformation matrix into solve_mat.
        Uses the formula: match_matrix * <a,b,c,d,e,f> = <x'[1], y'[1]...x'[n], y'[n]>
        Where all variables marked " ' " are the feature points in image2.
        The weights for the affine matrix are placed as shown:

        [a,b,c]
        [d,e,f]
        [0,0,1]

        This formula is solved using:  <a,b,c,d,e,f> = (match_matrix.T * match_matrix)^-1 * match_matrix.T * <x'[1], y'[1]...x'[n], y'[n]>'''
        match_matrix = self.get_match_matrix()
        features2_vec = self.get_destination_vector()

        '''
        not sure why this wasn't working. Is likely because I used too many bad features... Try again once I have implemented RANSAC
        affine_mat_weights = ((np.linalg.inv((match_matrix.T).dot(match_matrix))).dot(match_matrix.T)).dot(features2_vec)

        print("affine mat weights shape: ", affine_mat_weights.shape)

        affine_mat = np.array([[affine_mat_weights[0], affine_mat_weights[1], affine_mat_weights[2]],
                               [affine_mat_weights[3], affine_mat_weights[4], affine_mat_weights[5]],
                               [0,                     0,                     1.0]])
        print("affine mat: ", affine_mat)
        self.align_mat = affine_mat'''

        xy1s, xy2s = self.matches_to_points()
        cv_affine_mat = cv2.getAffineTransform(xy1s, xy2s)
        affine_mat = np.array([[cv_affine_mat[0,0], cv_affine_mat[0,1], cv_affine_mat[0,2]],
                               [cv_affine_mat[1,0], cv_affine_mat[1,1], cv_affine_mat[1,2]],
                               [0, 0, 1.0]])
        self.align_mat = affine_mat

    def matches_to_points(self):
        features1 = []
        features2 = []
        for i in range(0, len(self.solve_feature_matches)):
            features1.append(self.solve_feature_matches[i].xy1)
            features2.append(self.solve_feature_matches[i].xy2)
        features1 = np.float32(np.array(features1))
        features2 = np.float32(np.array(features2))
        return features1, features2


    '''creates a match matrix as shown in http://www.cs.cornell.edu/courses/cs4670/2016sp/lectures/lec16_alignment_web.pdf
    that allows this to be solveable easily by matrix algebra'''
    def get_match_matrix(self):
        match_matrix = np.zeros((2*len(self.solve_feature_matches), 6))
        for i in range(0, match_matrix.shape[0], 2):
            match_matrix[i] = np.array([self.solve_feature_matches[i//2].xy1[0], self.solve_feature_matches[i//2].xy1[1], 1.0, 0, 0, 0])
            match_matrix[i+1] = np.array([0, 0, 0, self.solve_feature_matches[i//2].xy1[0], self.solve_feature_matches[i//2].xy1[1], 1.0])
        return match_matrix
    '''creates a vector as shown meant to represent the features2 as shown in:
    http://www.cs.cornell.edu/courses/cs4670/2016sp/lectures/lec16_alignment_web.pdf'''
    def get_destination_vector(self):
        dest_vector = np.zeros((2*len(self.solve_feature_matches)))
        for i in range(0, dest_vector.shape[0], 2):
            dest_vector[i] = self.solve_feature_matches[i//2].xy2[0]
            dest_vector[i+1] = self.solve_feature_matches[i//2].xy2[1]
        return dest_vector

    def transform_image(self, image):
        untransformed_corner_vectors = [np.array([0,0,1.0]), np.array([image.shape[1], 0,1.0]), np.array([image.shape[1], image.shape[0],1.0]), np.array([0, image.shape[0],1.0])]

        cv_affine_align_mat = self.align_mat[:2, :].copy()
        transformed_corner_vectors = [self.align_mat.dot(untransformed_corner_vectors[i]) for i in range(0, len(untransformed_corner_vectors))]
        transformed_origin = transformed_corner_vectors[0]

        print('transformed corner vectors: ', transformed_corner_vectors)


        bounding_box = VectorMath.vectors_bounding_box(transformed_corner_vectors)
        transformed_image_dims = (int(bounding_box[2]-bounding_box[0]), int(bounding_box[3] - bounding_box[1]))
        cv_affine_align_mat[0,2] -= bounding_box[0]
        cv_affine_align_mat[1,2] -= bounding_box[1]








        transformed_corner_vectors = np.array([-bounding_box[0], -bounding_box[1]])
        '''issue with transformed_origin being incorrect is because opencv automatically aligns the image so that it will fit inside of the frame,
        causing the values to be inaccurate'''

        transformed_image = cv2.warpAffine(image, cv_affine_align_mat, transformed_image_dims)
        

        '''transformed_feature_matches = []
        for i in range(0, len(self.all_feature_matches)):
            transformed_feature_matches.append(self.all_feature_matches[i].transform(self))'''

        return transformed_image#, transformed_origin[:2]#, transformed_feature_matches


    '''def transform_image_points(self, image_points):
        transformed_points = []
        for i in range(0, len(image_points)):
            point_xy = np.array([image_points[i][0][0], image_points[i][0][1], 1.0])
            transformed_point = (self.align_mat.dot(point_xy))[0:2]
            transformed_points.append((transformed_point, image_points[i][1]))
        return transformed_points'''




    def transform_feature_match(self, feature_match):
        point_to_transform = np.array([feature_match.xy1[0], feature_match.xy1[1], 1.0])
        transformed_point = self.align_mat.dot(point_to_transform)
        image_point2d = transformed_point[:2]
        return image_point2d

    def NUM_SOLVE_FEATURES():
        return 3
