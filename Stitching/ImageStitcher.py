import numpy as np
'''takes the base image (the one being transformed), image1, the fit image (image2), and the AlignSolver object which will orient image1
to match image2'''
def stitch_images(image1, image2, align_solve):
    image1_corners = [ np.array([0,0]), np.array([0,image1.shape[1]]), np.array([image1.shape[0], image1.shape[1]]), np.array([image1.shape[0], 0]) ]
