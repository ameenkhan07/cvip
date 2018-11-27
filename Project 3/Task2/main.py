import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean

import os

OUTPUT_DIR = "outputs/"
img_name = "./point.jpg"


def _save(filename, img):
    """Saves the image with filename in output dir 
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filename = os.path.join(OUTPUT_DIR, filename)
    cv.imwrite(filename, img)


def _pad(img, pad):
    """Returns the padded image
    """
    dim_y, dim_x = img.shape
    pad_y, pad_x = pad
    padded_img = np.asarray([[0 for i in range(dim_x + (pad_x * 2))]
                             for j in range(dim_y + (pad_y * 2))],
                            dtype=np.float32)
    padded_img[pad_x:-pad_x, pad_y:-pad_y] = img
    return(padded_img)


def _point_detection(img, mask, thresh):
    """
    """
    dim_y, dim_x = img.shape
    res = np.asarray([[0 for _ in range(dim_x)] for _ in range(dim_y)],
                     dtype=np.uint8)
    res2 = np.asarray([[0 for _ in range(dim_x)] for _ in range(dim_y)],
                      dtype=np.uint8)
    # img = _pad(img, (3, 3))
    ptr = 1
    for x in range(1, dim_y - 1):
        for y in range(1, dim_x - 1):
            temp = abs(sum(sum(img[x - 1:x + 2, y - 1:y + 2] * mask)))
            if temp > thresh:
                res2[x][y] = 255
                print(f'Point {ptr} : ({x}, {y})')
                ptr += 1
            else:
                res2[x][y] = 0
            res[x - 1, y - 1] = temp
    _save('masked_image.png', res)
    _save('point_detection.png', res2)


if __name__ == '__main__':

    # Part A : Point detection algoritm
    img = cv.imread(img_name, 0)
    mask = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    thresh = threshold_mean(img)
    _point_detection(img, mask, thresh)
