import cv2 as cv
import numpy as np
import os

# img = cv.imread('./task3/pos_3.jpg',0)
# img2 = img.copy()
template = cv.imread('./task3/template_alt_alt.jpg', 0)
template_laplacian = cv.Laplacian(template, cv.CV_8U)
w, h = template_laplacian.shape[::-1]


# All the 6 methods for comparison in a list
methods = [
    'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED',
    'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
]

images = [
    'pos_1.jpg', 'pos_2.jpg', 'pos_3.jpg', 'pos_4.jpg', 'pos_5.jpg',
    'pos_6.jpg', 'pos_7.jpg', 'pos_8.jpg', 'pos_9.jpg', 'pos_10.jpg',
    'pos_11.jpg', 'pos_12.jpg', 'pos_13.jpg', 'pos_14.jpg', 'pos_15.jpg'
]

# meth = 'cv.TM_CCORR_NORMED'

for meth in methods:
    if not os.path.exists(meth):
        os.makedirs('./outputs/' + meth)
    for image in images:
        img = cv.imread('./task3/' + image, 0)
        img_blur = cv.GaussianBlur(img, (3, 3), 0)
        img_blur_laplacian = cv.Laplacian(img_blur, cv.CV_8U)
        
        method = eval(meth)
        
        # Apply template Matching
        # res = cv.matchTemplate(img_blur_laplacian, template_laplacian, method)
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        cv.imwrite('./outputs/' + meth + '/' + image, img)
