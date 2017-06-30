import cv2
import numpy as np

def get_image_gradients(image, dx, dy):
    grad_x = cv2.Sobel(image, cv2.CV_32F, dx, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, dy)
    return grad_x, grad_y

def get_gradient_mags(grad_x, grad_y):
    return np.sqrt(grad_x**2 + grad_y**2)

def get_phase_image(grad_x, grad_y):
    return cv2.phase(grad_x, grad_y)
