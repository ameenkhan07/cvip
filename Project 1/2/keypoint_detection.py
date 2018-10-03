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

OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class ScaleSpace:
    """ Building Scale Space
    and calculating DoG
    """

    def __init__(self):
        self.sigma = 1 / math.sqrt(2)
        self.k = math.sqrt(2)
        self.gauss_pyramid = []  # array of all octaves
        self.dog = []  # array of all DoGs of all octaves
        self.dog_extrema = []
        self.img = cv2.imread("./task2.jpg", 0)  # Greyscaled Image
        self.original_img = cv2.imread("./task2.jpg")

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

        for i, octave in enumerate(self.dog):
            for im, scale in enumerate(octave):
                name = "DOG_" + str(i + 1) + "_" + str(im + 1) + ".png"
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, name),
                    np.asarray(scale, dtype=np.uint8))
                # save_img(np.asarray(scale, dtype=np.uint8), name)
                # self._show_img(im)

    def _detect_keypoints(self):
        """Save images for all extrema for all Octave DoGs
        """
        for o, octave in enumerate(self.dog):
            extremas_ = []
            for s, scale in enumerate(octave):
                if s in [1, 2]:
                    DX, DY = scale.shape[0], scale.shape[1]
                    temp = [[0 for _ in range(DY)] for _ in range(DX)]
                    for x, row in enumerate(octave[s], start=0):
                        for y, col in enumerate(row, start=0):
                            neighborhood = get_neigborhood(
                                x, y, scale, octave[s - 1], octave[s + 1])
                            if col < min(neighborhood) or col > max(
                                    neighborhood):
                                temp[x][y] = 255
                    extremas_.append(temp)
                    name = "Extrema_" + str(o + 1) + "_" + str(s + 1) + ".png"
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, name),
                        np.asarray(temp, dtype=np.uint8))
            self.dog_extrema.append(extremas_)

    def _save_keypoint_overlay(self):
        """
        """
        oct_img = self._sample_img(self.original_img)

        # For every sampled image corresponding to a octave
        for s, s_img in enumerate(oct_img):
            for o, extrema_img in enumerate(self.dog_extrema[s]):
                DX, DY = s_img.shape[0], s_img.shape[1]
                # print(DX, DY)
                for i in range(DX):
                    for j in range(DY):
                        if extrema_img[i][j] == 255:
                            s_img[i][j] = 255
                # self._show_img(s_img)
                name = "Octave_Images_" + str(s + 1) + "_" + str(o +
                                                                 1) + ".png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, name), s_img)


s = ScaleSpace()
s._create_gauss_pyramid()
s._create_diff_of_gauss()
s._detect_keypoints()
s._save_keypoint_overlay()