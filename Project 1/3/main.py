import cv2 as cv
import numpy as np
import os

template = cv.imread('./task3/template_alt_alt.jpg', 0)
template_laplacian = cv.Laplacian(template, cv.CV_8U)
w, h = template_laplacian.shape[::-1]

images = [
    'pos_1.jpg',
    'pos_2.jpg',
    'pos_3.jpg',
    'pos_4.jpg',
    'pos_5.jpg',
    'pos_6.jpg',
    'pos_7.jpg',
    'pos_9.jpg',
    'pos_10.jpg',
    'pos_11.jpg',
    'pos_12.jpg',
    'pos_13.jpg',
    'pos_14.jpg',
    'pos_15.jpg',
    'neg_1.jpg',
    'neg_2.jpg',
    'neg_3.jpg',
    'neg_4.jpg',
    'neg_5.jpg',
    'neg_6.jpg',
    'neg_8.jpg',
    'neg_9.jpg',
    'neg_10.jpg',
]

method = eval('cv.TM_CCORR_NORMED')
OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for image in images:

    img = cv.imread('./task3/' + image, 0)
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    img_blur_laplacian = cv.Laplacian(img_blur, cv.CV_8U)

    # print(type(method))
    # method = eval(method)

    # Apply template Matching
    res = cv.matchTemplate(img_blur_laplacian, template_laplacian, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    cv.imwrite(os.path.join(OUTPUT_DIR, image), img)
