# Keypoint Detection
# Detect keypoints in an image according to the following steps, which are also the first three steps of Scale-Invariant Feature Transform (SIFT).
# 1.  Generate four octaves.  Each octave is composed of five images blurred using Gaussian kernels.  For each octave, the bandwidth parameters σ(five different scales) of the Gaussian kernels are shown in Tab.  1.
# 2.  Compute Difference of Gaussian (DoG) for all four octaves.
# 3.  Detect keypoints which are located at the maxima or minima of the DoG images.  You only need to provide pixel-level locations of the keypoints; you do not need to provide sub-pixel-level locations.

# In your report, please
# (1) include images of the second and third octave and specify their resolution (width×height, unit pixel);
# (2) include DoG images obtained using the second and third octave;
# (3) clearly show all the detected keypoints using white dots on the original image
# (4) provide coordinates of the five left-most detectedkeypoints (the origin is set to be the top-left corner).

import cv2
import math
import numpy as np
import scipy.ndimage
from utils import *


class ScaleSpace:
    """ Building Scale Space
    and calculating DoG
    """

    def __init__(self):
        self.sigma = 1 / math.sqrt(2)
        self.k = math.sqrt(2)
        self.gauss_pyramid = []  # array of all octaves
        self.dog = []  # array of all DoGs of all octaves
        self.img = cv2.imread("./task2.jpg", 0)

    def _show_img(self, img, name='IMAGE'):
        """Utility Function to cv2 show images
        """
        cv2.imshow(name, np.asarray(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _scale_space(self, SIG, img):
        """Returns a list of image arrays for a single octave
        Parameters :
            SIG : int, starting value for sigma
        Return:
            octave : List, list of image arrays for a single octave
        """
        # SIG = 1/math.sqrt(2)
        k = math.sqrt(2)
        sig = [SIG, k * SIG, k * k * SIG, k * k * k * SIG, k * k * k * k * SIG]

        # temp = signal.convolve2d(img,_gaussian_kernel(SIG))

        octave_list = []
        for s in sig:
            temp = scipy.ndimage.filters.gaussian_filter(img, s)
            octave_list.append(temp)
            # self._show_img(temp)

        return octave_list

    def _create_gauss_pyramid(self):
        """Generates gaussian images for 4 octaves
        """
        SIG = [
            1 / math.sqrt(2),
            math.sqrt(2), 2 * math.sqrt(2), 4 * math.sqrt(2)
        ]

        # List of all sampled images
        oct_img = self._sample_img(self.img)
        for ele in zip(SIG, oct_img):
            # print(ele[0], ele[1].shape)
            octave = self._scale_space(ele[0], ele[1])
            self.gauss_pyramid.append(octave)

    def _sample_img(self, img):
        """Resizes images to be used for different octaves
        Parameter:
            oct_img: List of image
        Return
        """
        oct_img = [img]
        for _ in range(3):
            oct_img.append(oct_img[-1][::2, ::2])
        return oct_img

    def _create_diff_of_gauss(self):
        """Creates complete DoG for all octaves
        """
        for oct in self.gauss_pyramid:
            self.dog.append([y - x for x, y in zip(oct, oct[1:])])

        for i in self.dog:
            for im in i:
                self._show_img(im)
        for i, octave in enumerate(self.dog):
            for im, scale in enumerate(octave):
                cv2.imwrite("DOG_"+str(i)+"_"+str(im)+".png", np.asarray(scale, dtype = np.uint8))
                # self._show_img(im)


s = ScaleSpace()
s._create_gauss_pyramid()
s._create_diff_of_gauss()