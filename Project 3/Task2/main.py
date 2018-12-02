import cv2 as cv
import numpy as np
import os

OUTPUT_DIR = "outputs/"
img_name_1 = "./point.jpg"
img_name_2 = "./segment.jpg"


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

    res = np.zeros((dim_y, dim_x))
    res2 = np.zeros((dim_y, dim_x))

    for i in range(dim_y - 2):
        for j in range(dim_x - 2):
            res[i][j] = abs(sum(sum(img[i:i+3, j: j+3] * mask)))
            if res[i][j] > thresh:
                res2[i][j] = 255

    # Save Images after pask and thresholding
    _save('masked_image.jpg', res)
    _save('point_detection.jpg', res2)



if __name__ == '__main__':

    # Part A : Point detection algoritm

    img1 = cv.imread(img_name_1, 0)
    # Negative Laplacian Filter
    mask = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Threshold the image to get the points, using the mask
    _point_detection(img1, mask, 314)

