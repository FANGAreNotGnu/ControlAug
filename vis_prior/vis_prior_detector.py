import cv2 as cv
from PIL import ImageFilter

import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../../ControlNet"))
from annotator.hed import HEDdetector
from annotator.mlsd import MLSDdetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector

from .utils import *

PIL_FILTERS = {
    "BLUR": ImageFilter.BLUR,
    "CONTOUR": ImageFilter.CONTOUR, 
    "DETAIL": ImageFilter.DETAIL, 
    "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE, 
    "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE, 
    "EMBOSS": ImageFilter.EMBOSS, 
    "FIND_EDGES": ImageFilter.FIND_EDGES, 
    "SHARPEN": ImageFilter.SHARPEN, 
    "SMOOTH": ImageFilter.SMOOTH, 
    "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
}  # https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html

class PILEdgeDetector():
    def __call__(self, img, modes):
        return self.detect(img=img, modes=modes)

    def detect(self, im, modes):
        im1 = copy_numpy_to_pil(im)
        for mode in modes:
            if mode in PIL_FILTERS:
                im1 = im1.filter(PIL_FILTERS[mode])
            else:
                raise ValueError(f"{mode} is not supported as EdgeDetector's mode")
        return copy_pilrgb_to_cv(im1)

class CannyEdgeDetector():
    def __init__(self, low=100, high=200, blur=0):
        self.low = low
        self.high = high
        self.blur = blur

    def __call__(self, img, low=None, high=None, blur=None):
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        if blur is None:
            blur = self.blur
        return self.detect(img=img, low=low, high=high, blur=blur)

    def detect(self, img, low, high, blur):
        #print(img.shape)
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (blur, blur), 0)
        return cv.Canny(img, low, high)


class HEDEdgeDetector():
    def __init__(self):
        self.model = HEDdetector()
    def __call__(self, img):
        return self.model(img)


class ScribbleEdgeDetector():
    def __init__(self):
        self.model = HEDdetector()
    
    def __call__(self, img):
        detected_map = self.model(img)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        return detected_map


class UniformerMaskDetector():
    def __init__(self):
        self.model = UniformerDetector()
    def __call__(self, img):
        return self.model(img)
