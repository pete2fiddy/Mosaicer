from Feature.MatchType import MatchType
import numpy as np
from Feature.FeatureMatch import FeatureMatch
import cv2
from matplotlib import pyplot as plt
import ImageOp.CVMath as CVMath
import ImageOp.ImageMath as ImageMath
from PIL import Image
from math import pi, sqrt
import ImageOp.HOG as HOG

class HOGMatch(MatchType):
    GAUSSIAN_BLUR_WINDOW = (5,5)
    GAUSSIAN_BLUR_STD_DEV = 1.0
    DEFAULT_SMALL_WINDOW = 8
    DEFAULT_NUM_AGGREGATE_WINDOWS = 4
    NUM_BINS = 9
    THETA_PER_INDEX = pi/float(NUM_BINS)

    '''should make this take a NamedArgs object in the future'''
    def __init__(self, image1, image2, mask, small_window = None, num_aggregate_windows = None):
        self.small_window = small_window if small_window is not None else HOGMatch.DEFAULT_SMALL_WINDOW
        self.num_aggregate_windows = num_aggregate_windows if num_aggregate_windows is not None else HOGMatch.DEFAULT_NUM_AGGREGATE_WINDOWS
        self.aggregate_window_size = int(sqrt(self.num_aggregate_windows))
        MatchType.__init__(self, image1, image2, mask)


    def init_features(self):
        self.init_hog_maps()
        image1_keypoints, self.image1_descriptors = self.hog_map_to_keypoints_and_descriptors(self.hogs1)
        image2_keypoints, self.image2_descriptors = self.hog_map_to_keypoints_and_descriptors(self.hogs2)
        self.set_features(image1_keypoints, image2_keypoints)

    def match_features(self):
        '''Not sure what the paramaters for BFMatcher below do, but docs said
        they were recommended for ORB (not sure about HOG)'''
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        kp_matches = bf_matcher.match(self.image1_descriptors, self.image2_descriptors)
        out_image = np.zeros((self.image1.shape[0] + self.image2.shape[0], self.image1.shape[1], 3))
        match_image = cv2.drawMatches(self.image1, self.features1, self.image2, self.features2, kp_matches, out_image)


        feature_matches = FeatureMatch.cv_matches_to_feature_matches(kp_matches, self.features1, self.features2)
        self.set_matches(feature_matches)

    def init_hog_maps(self):
        blur_image1 = cv2.GaussianBlur(cv2.cvtColor(self.image1, cv2.COLOR_RGB2GRAY), HOGMatch.GAUSSIAN_BLUR_WINDOW, HOGMatch.GAUSSIAN_BLUR_STD_DEV)
        blur_image2 = cv2.GaussianBlur(cv2.cvtColor(self.image2, cv2.COLOR_RGB2GRAY), HOGMatch.GAUSSIAN_BLUR_WINDOW, HOGMatch.GAUSSIAN_BLUR_STD_DEV)
        grad_x1, grad_y1 = CVMath.get_image_gradients(blur_image1, 1, 1)
        mags1 = CVMath.get_gradient_mags(grad_x1, grad_y1)
        phase1 = np.mod(CVMath.get_phase_image(grad_x1, grad_y1), pi)
        grad_x2, grad_y2 = CVMath.get_image_gradients(blur_image2, 1, 1)
        mags2 = CVMath.get_gradient_mags(grad_x2, grad_y2)
        phase2 = np.mod(CVMath.get_phase_image(grad_x2, grad_y2), pi)

        small_hogs1 = self.create_small_hog_map(phase1, mags1)
        small_hogs2 = self.create_small_hog_map(phase2, mags2)

        self.hogs1 = self.create_large_hog_map(small_hogs1)
        self.hogs2 = self.create_large_hog_map(small_hogs2)

    '''creates an un-normalized gradient magnitude and angle histogram using small
    self.small_window x self.small_window size windows. Output is an array that corresponds
    to the [i,j]th small window'''
    def create_small_hog_map(self, phase, mags):
        small_hogs = np.zeros((mags.shape[0]//self.small_window, mags.shape[1]//self.small_window, HOGMatch.NUM_BINS))
        for x in range(0, small_hogs.shape[0]):
            for y in range(0, small_hogs.shape[1]):
                phase_window = phase[x * self.small_window : (x+1) * self.small_window, y * self.small_window : (y+1) * self.small_window]
                mags_window = mags[x * self.small_window : (x+1) * self.small_window, y * self.small_window : (y+1) * self.small_window]
                windowed_hist = HOG.HOG_window(phase_window, mags_window, HOGMatch.NUM_BINS)#self.create_windowed_histogram(phase_window, mags_window)
                small_hogs[x,y] = windowed_hist
        return small_hogs

    '''
    def create_windowed_histogram(self, phase_window, mags_window):
        hist = np.zeros((HOGMatch.NUM_BINS))
        flat_phases = phase_window.flatten()
        flat_mags = mags_window.flatten()
        for i in range(0, flat_phases.shape[0]):
            lower_hist_index = int(flat_phases[i]/HOGMatch.THETA_PER_INDEX)
            upper_hist_index = lower_hist_index + 1 if lower_hist_index < hist.shape[0] - 1 else 0
            proportion_to_lower_index = (flat_phases[i] - (HOGMatch.THETA_PER_INDEX * lower_hist_index))/HOGMatch.THETA_PER_INDEX
            proportion_to_upper_index = 1.0 - proportion_to_lower_index
            hist[lower_hist_index] += proportion_to_lower_index * float(flat_mags[i])
            hist[upper_hist_index] += proportion_to_upper_index * float(flat_mags[i])
        return hist
    '''
    '''returns a normalized small_hogs.shape[0] x small_hogs.shape[1] x NUM_BINS * num_aggregate_windows matrix. For the
    index at [i,j], the vector has concetanated the small hogs into a longer vector of length num_aggregate_windows*num_bins
    Each set of n*num_bins to (n+1)*num_bins describes a single HOG vector (i.e. all indexes that mod by num_bins to equal the
    same value belong to the same angle)'''
    def create_large_hog_map(self, small_hogs):
        big_hogs = np.zeros((small_hogs.shape[0], small_hogs.shape[1], HOGMatch.NUM_BINS * self.num_aggregate_windows))

        for x in range(0, big_hogs.shape[0] - self.aggregate_window_size + 1):
            for y in range(0, big_hogs.shape[1] - self.aggregate_window_size + 1):
                small_hogs_window = small_hogs[x:x+self.aggregate_window_size, y:y+self.aggregate_window_size]
                window_hog_vector = small_hogs_window.flatten()
                big_hogs[x,y] = window_hog_vector/np.linalg.norm(window_hog_vector)
        return big_hogs

    '''for each index of the hog map, creates a keypoint point centered on the window's center
    where it is placed on the image. Also returns the HOG descriptors for each keypoint'''
    def hog_map_to_keypoints_and_descriptors(self, hog_map):
        kps = []
        descriptors = []
        window_margin = self.small_window//2
        for x in range(0, hog_map.shape[0]):
            for y in range(0, hog_map.shape[1]):
                append_kp = cv2.KeyPoint((y*self.small_window) + window_margin, (x*self.small_window) + window_margin, self.small_window)
                kps.append(append_kp)
                descriptors.append(hog_map[x,y])
        descriptors = np.asarray(descriptors)
        '''opencv's descriptor matching method requires the vectors to be uint8'''
        descriptors *= 255
        descriptors = descriptors.astype(np.uint8)
        return kps, descriptors

    '''takes the list of descriptors and attempts to transform them so to allow HOG to be
    rotationally invarient. Globally transforms all HOGS in image1 to best fit the hogs of
    image2. (Note, may be best to actually do this on a per-case basis, i.e. when two descriptors
    are being compared for match quality, transform one descriptor to optimally fit the second.
    The drawbacks to this would be that the rotation would not be robust to noise, i.e. the rotation
    necessary to apply to each HOG is likely largely uniform since if the image were rotated,
    the entire image would be rotated). Will likely not be robust to images with very high rotation (> 90 degrees?)
    '''
    def fit_descriptors(self):
        return None
