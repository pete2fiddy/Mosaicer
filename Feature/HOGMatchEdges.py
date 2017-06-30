from Feature.MatchType import MatchType
import numpy as np
from Feature.FeatureMatch import FeatureMatch
import cv2
import ImageOp.CVMath as CVMath
import ImageOp.ImageMath as ImageMath
from PIL import Image
from math import pi, sqrt
import ImageOp.HOG as HOG

class HOGMatchEdges(MatchType):
    GAUSSIAN_BLUR_WINDOW = (7,7)#(7,7)
    GAUSSIAN_BLUR_STD_DEV = 1.5
    DEFAULT_CANNY_THRESH = (160, 200)#(400, 700)#(400, 500)#(160, 200)
    DEFAULT_HOG_WINDOW = 5
    DEFAULT_BINS = 18
    DEFAULT_MAX_MATCH_SCORE = 100#350
    '''should probably use corners instead of edges to 1: speed it up, and 2:
    prevent feature generalization (i.e. parking lines will have almost identical gradients
    to each other because they are all parallel, straight lines. At least corners would account
    for more interesting outside gradients)'''

    '''should make this take a NamedArgs object in the future'''
    def __init__(self, image1, image2, mask, hog_window = None, bins = None, canny_thresh = None, max_match_score = None):
        self.image1 = image1
        self.image2 = image2
        self.mask = mask
        self.max_match_score = max_match_score if max_match_score is not None else HOGMatchEdges.DEFAULT_MAX_MATCH_SCORE
        self.hog_window = hog_window if hog_window is not None else HOGMatchEdges.DEFAULT_HOG_WINDOW
        self.hog_window_margin = (self.hog_window - 1)//2
        self.bins = bins if bins is not None else HOGMatchEdges.DEFAULT_BINS
        self.canny_thresh = canny_thresh if canny_thresh is not None else HOGMatchEdges.DEFAULT_CANNY_THRESH
        MatchType.__init__(self, image1, image2, mask)

    def init_features(self):
        mags1, phase1, canny1 = self.get_sobel_phase_and_canny_images(self.image1)


        keypoints1, self.hogs1 = self.create_keypoints_and_hog_descriptors(mags1, phase1, canny1)
        hog_image1 = self.create_hog_map_image(keypoints1, self.hogs1, mags1.shape[:2])
        Image.fromarray(hog_image1).show()

        mags2, phase2, canny2 = self.get_sobel_phase_and_canny_images(self.image2)

        keypoints2, self.hogs2 = self.create_keypoints_and_hog_descriptors(mags2, phase2, canny2)



        hog_image2 = self.create_hog_map_image(keypoints2, self.hogs2, mags2.shape[:2])
        Image.fromarray(hog_image2).show()

        self.set_features(keypoints1, keypoints2)


    def match_features(self):
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        kp_matches = bf_matcher.match(self.hogs1, self.hogs2)
        out_image = np.zeros((self.image1.shape[0] + self.image2.shape[0], self.image1.shape[1], 3))

        temp_kp_matches = []
        for i in range(0, len(kp_matches)):
            index1 = kp_matches[i].queryIdx
            index2 = kp_matches[i].trainIdx
            #cum_trapz1 = np.array([np.trapz(self.hogs1[index1][:j]) for j in range(0, self.hogs1[index1].shape[0])])
            #cum_trapz2 = np.array([np.trapz(self.hogs2[index2][:j]) for j in range(0, self.hogs2[index2].shape[0])])

            hog_score = np.linalg.norm(self.hogs1[index1] - self.hogs2[index2])#sqrt(np.sum((cum_trapz2 - cum_trapz1)**2))#
            if i % 50 == 0:
                print("hog score: ", hog_score)
            if hog_score < self.max_match_score:
                temp_kp_matches.append(kp_matches[i])
        kp_matches = temp_kp_matches

        match_image = cv2.drawMatches(self.image1, self.features1, self.image2, self.features2, kp_matches, out_image)
        Image.fromarray(match_image).show()



        feature_matches = FeatureMatch.cv_matches_to_feature_matches(kp_matches, self.features1, self.features2)



        self.set_matches(feature_matches)

    def create_keypoints_and_hog_descriptors(self, mags, phase, canny):
        hog_indexes = np.where(canny > 0)
        hogs = []
        kps = []
        for i in range(0, hog_indexes[0].shape[0]):
            hog_x = hog_indexes[0][i]
            hog_y = hog_indexes[1][i]
            if hog_x > self.hog_window_margin and hog_x < mags.shape[0] - self.hog_window_margin:
                if hog_y > self.hog_window_margin and hog_y < mags.shape[1] - self.hog_window_margin:
                    phase_window = phase[hog_x - self.hog_window_margin : hog_x + self.hog_window_margin + 1, hog_y - self.hog_window_margin : hog_y + self.hog_window_margin + 1]
                    mags_window = mags[hog_x - self.hog_window_margin : hog_x + self.hog_window_margin + 1, hog_y - self.hog_window_margin : hog_y + self.hog_window_margin + 1]
                    hist = HOG.HOG_window(phase_window, mags_window, self.bins)
                    hist /= np.linalg.norm(hist)
                    hogs.append(hist)
                    kps.append(cv2.KeyPoint(hog_y, hog_x, self.hog_window))
        hogs = np.asarray(hogs)
        '''opencv's matching functions require the descriptors to be uint8. The below
        converts to uint8 (with hopefully little lossiness)'''
        hogs = (255 * hogs).astype(np.uint8)
        return kps, hogs

    '''purely for visualizing hog. Attempts to divide the bins evenly into the RGB
    channels of the image'''
    def create_hog_map_image(self, keypoints, hogs, image_dims):
        out_image = np.zeros((image_dims[0], image_dims[1], 3))
        for i in range(0, len(keypoints)):
            kp_x = keypoints[i].pt[0]
            kp_y = keypoints[i].pt[1]
            '''keypoints require absolute x and y so they are flipped to be mapped
            to the image correctly'''
            hist = hogs[i]
            rgb_splits = np.array_split(hist, 3)
            #print("rgb splits: ", rgb_splits)
            r = int(np.linalg.norm(rgb_splits[0]))
            g = int(np.linalg.norm(rgb_splits[1]))
            b = int(np.linalg.norm(rgb_splits[2]))
            out_image[int(kp_y), int(kp_x)] = np.array([r,g,b], dtype = np.uint8)
        return np.uint8(out_image)


    '''returns a sobel magnitude and phase image for the inputted color image'''
    def get_sobel_phase_and_canny_images(self, image):
        '''might need to convert to RGB from BGR. Not a huge deal if it is flipped
        (as green is in the middle either way and it is the most highly waited in conversion
        to grayscale)'''
        bw_blurred_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bw_blurred_image = cv2.GaussianBlur(bw_blurred_image, HOGMatchEdges.GAUSSIAN_BLUR_WINDOW, HOGMatchEdges.GAUSSIAN_BLUR_STD_DEV)
        gradx, grady = CVMath.get_image_gradients(bw_blurred_image, 1, 1)
        mags = CVMath.get_gradient_mags(gradx, grady)
        phase = np.mod(CVMath.get_phase_image(gradx, grady), pi)
        canny = cv2.Canny(bw_blurred_image, self.canny_thresh[0], self.canny_thresh[1])
        return mags, phase, canny
