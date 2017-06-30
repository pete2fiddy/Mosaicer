from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import pi
import ImageOp.ImageMath as ImageMath
import Function.Smooth1D as Smooth1D
import Function.Func1D as Func1D
import timeit

class FullMatchMosaicer:
    GAUSSIAN_BLUR_WINDOW = (5,5)
    GAUSSIAN_BLUR_STD_DEV = 2.0
    RESIZE_DIMS = (1600,1200)
    MIN_SOBEL_MAG = 100
    NUM_FULL_HIST_BINS = 120
    def __init__(self, totrans, identity):
        self.identity = cv2.resize(identity, FullMatchMosaicer.RESIZE_DIMS)
        self.totrans = cv2.resize(totrans, FullMatchMosaicer.RESIZE_DIMS)


        self.identity = cv2.cvtColor(self.identity, cv2.COLOR_BGR2GRAY)
        self.totrans = cv2.cvtColor(self.totrans, cv2.COLOR_BGR2GRAY)



        blurred_identity = cv2.GaussianBlur(self.identity, FullMatchMosaicer.GAUSSIAN_BLUR_WINDOW, FullMatchMosaicer.GAUSSIAN_BLUR_STD_DEV)
        blurred_totrans = cv2.GaussianBlur(self.totrans, FullMatchMosaicer.GAUSSIAN_BLUR_WINDOW, FullMatchMosaicer.GAUSSIAN_BLUR_STD_DEV)

        iden_gradx, iden_grady = self.get_gradients(self.identity, 1, 1)
        iden_grad_mags = self.get_gradient_mags(iden_gradx, iden_grady)
        iden_phase = self.get_phase_image(iden_grad_mags, iden_gradx, iden_grady)
        iden_full_grad_hist = self.create_grad_hist(iden_phase, iden_grad_mags, FullMatchMosaicer.MIN_SOBEL_MAG, bins = FullMatchMosaicer.NUM_FULL_HIST_BINS)
        smooth_iden_full_grad_hist = Smooth1D.moving_avg_smooth(iden_full_grad_hist, 5, 2)
        smooth_iden_full_grad_hist = Func1D.norm_by_area(smooth_iden_full_grad_hist)

        totrans_gradx, totrans_grady = self.get_gradients(self.totrans, 1, 1)
        totrans_grad_mags = self.get_gradient_mags(totrans_gradx, totrans_grady)
        totrans_phase = self.get_phase_image(totrans_grad_mags, totrans_gradx, totrans_grady)
        totrans_full_grad_hist = self.create_grad_hist(totrans_phase, totrans_grad_mags, FullMatchMosaicer.MIN_SOBEL_MAG, bins =FullMatchMosaicer.NUM_FULL_HIST_BINS)
        smooth_totrans_full_grad_hist = Smooth1D.moving_avg_smooth(totrans_full_grad_hist, 5, 2)
        smooth_totrans_full_grad_hist = Func1D.norm_by_area(smooth_totrans_full_grad_hist)
        x_labels = [i * pi/float(FullMatchMosaicer.NUM_FULL_HIST_BINS) for i in range(0, FullMatchMosaicer.NUM_FULL_HIST_BINS)]


        plt.polar(x_labels, smooth_totrans_full_grad_hist, color = 'r')
        plt.polar(x_labels,smooth_iden_full_grad_hist,color = 'g')
        #plt.polar(x_labels, smooth_totrans_full_grad_hist - smooth_iden_full_grad_hist, color = 'b')
        plt.show()

    '''notes for a hog descriptor matcher:
    step 1: sort angle bins by magnitude of vote, keeping track of what bin is at what place after sorting
    (do for both distributions)
    step 2: subtract the distributions, multiplying each index of distraction by the angle between the two unit vectors pointing at that angle / 180 (to norm it)
    step 3: square each index after performing the above step
    step 4: sum the above. That is the error
    '''

    def get_gradients(self, image, dx, dy):
        grad_x = cv2.Sobel(image, cv2.CV_32F, dx, 0)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, dy)
        return grad_x, grad_y

    def get_gradient_mags(self, grad_x, grad_y):
        return np.sqrt(grad_x**2 + grad_y**2)

    def get_phase_image(self, grad_mags, grad_x, grad_y):
        phase_image = cv2.phase(grad_x, grad_y)
        return phase_image

    def create_grad_hist(self, angle_image, grad_mags, min_sobel_mag, bins = 120, norm = True):
        hist = np.zeros((bins))
        angles = angle_image.flatten()
        mags = grad_mags.flatten()
        print("angles shape: ", angles.shape)
        theta_per_index = (pi)/float(bins)


        angles_modded_by_pi = np.mod(angles, pi)
        for i in range(0, hist.shape[0]):
            hist[i] = ((angles_modded_by_pi >= theta_per_index * i) & (angles_modded_by_pi < theta_per_index * (i+1)) & (mags > min_sobel_mag)).sum()
        if norm:
            hist = Func1D.norm_by_area(hist)
        return hist
