from Toolbox.NamedArgs import NamedArgs

class MosaicMaker:
    '''creates a single mosaic image out of the two inputted images using the inputted parameters, feature selection types, etc.'''
    def __init__(self, image1, image2, match_type, align_solve_type, blend_func, mask = None):
        self.image1 = image1
        self.image2 = image2
        self.mask = mask
        self.match_type = match_type(image1, image2, self.mask)
        self.align_solve_type = align_solve_type
        self.blend_func = blend_func

    def create_mosaic(self, ransac_params, blend_params):
        stitch1, stitch2 = self.create_stitch_images(ransac_params)
        blended_image = self.blend_func(stitch1, stitch2, blend_params)
        return blended_image

    def create_stitch_images(self, ransac_params):
        stitch_image1, stitch_image2 = self.match_type.create_mosaic(self.align_solve_type, ransac_params)
        return stitch_image1, stitch_image2
