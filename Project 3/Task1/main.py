import cv2 as cv
import numpy as np
import os

OUTPUT_DIR = "outputs/"
img_name = "./noise.jpg"


def _save(filename, img):
    """Saves the image with filename in output dir 
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # filename = filename+'.png'
    filename = os.path.join(OUTPUT_DIR, filename)
    cv.imwrite(filename, img)


def _dilate(img, str_img=[]):
    '''Expands the poindary of img
    '''
    w, h = img.shape

    res = [[0 for _ in range(h)] for _ in range(w)]
    for i, im_row in enumerate(img):
        for j, im_ele in enumerate(im_row):
            if im_ele:
                for k, str_row in enumerate(str_img):
                    for l, str_ele in enumerate(str_row):
                        if (i+k >= w) or (j+l >= h):
                            continue
                        if str_ele:
                            res[i+k][j+l] = 255

    return(np.asarray(res))


def _erode(img, str_img=[]):
    '''Contracts the boundary of img
    '''
    w, h = img.shape
    se_w, se_h = str_img.shape
    res = [[0 for _ in range(h)] for _ in range(w)]
    for i, im_row in enumerate(img):
        for j, im_ele in enumerate(im_row):
            flag = True
            for k, str_row in enumerate(str_img):
                for l, str_ele in enumerate(str_row):
                    if (i+k >= w) or (j+l >= h):
                        continue
                    if str_ele != 0 and img[i+k][j+l] == 0:
                        flag = False
            if flag:
                res[i][j] = 255
    return(np.asarray(res))

def opening(img, str_img):
    """
    """
    res = _erode(img, str_img)
    res = _dilate(res, str_img)
    return res

def closing(img, str_img):
    """
    """
    res = _dilate(img, str_img)
    res = _erode(res, str_img)
    return res

if __name__ == '__main__':
    img = cv.imread(img_name, 0)

    str_img = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    res1 = _dilate(img, str_img)
    _save('dilate.png', res1)
    res2 = _erode(img, str_img)
    _save('erode.png', res2)
    print(res1.shape, res2.shape)

    res1 = opening(img, str_img)
    _save('opening.png', res1)
    res2 = closing(img, str_img)
    _save('closing.png', res2)
    print(res1.shape, res2.shape)

    # Denoising
    # Approach 1 : Opening then Closing
    temp1 = opening(img, str_img)
    res1 = closing(temp1, str_img)
    _save('res_noise1.png', res1)

    # Approach 1 : Closing then Opening
    temp2 = closing(img, str_img)
    res2 = opening(temp2, str_img)
    _save('res_noise2.png', res2)
