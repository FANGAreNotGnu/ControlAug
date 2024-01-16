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
    def __init__(self, low=100, high=200, blur=5):
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

    def detect(self, img, low=100, high=200, blur=5):
        #print(img.shape)
        #img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.GaussianBlur(img, (blur, blur), 0)
        return cv.Canny(img, low, high)


class HEDEdgeDetector():
    def __init__(self):
        self.model = HEDdetector()
    def __call__(self, img):
        return self.model(img)


class MLSDEdgeDetector():
    def __init__(self, thr_v=0.1, thr_d=0.1):
        self.model = MLSDdetector()
        self.thr_v = thr_v
        self.thr_d = thr_d
    def __call__(self, img, thr_v=0.1, thr_d=0.1):
        # thr_v: value threshold
        # thr_d: distance threshold
        if thr_v is None:
            thr_v = self.thr_v
        if thr_d is None:
            thr_d = self.thr_d
        return self.model(img, thr_v, thr_d)


class MidasDepthDetector():
    def __init__(self, a=6.2):
        self.model = MidasDetector()
        self.a = a
    def __call__(self, img, a=6.2):
        # a: alpha
        if a is None:
            a = self.a
        return self.model(img, a)


class UniformerMaskDetector():
    def __init__(self):
        self.model = UniformerDetector()
    def __call__(self, img):
        return self.model(img)
