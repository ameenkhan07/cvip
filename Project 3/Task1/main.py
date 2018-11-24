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
    '''
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

if __name__ == '__main__':
    img = cv.imread(img_name, 0)
    
    str_img = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # print(img.shape, str_img.shape)

    res = _dilate(img, str_img)
    # print(res.shape)
    _save('temp.png', res)
    