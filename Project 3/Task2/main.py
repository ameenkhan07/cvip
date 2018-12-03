import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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


def _point_detection(img, mask, thresh):
    """
    """
    dim_y, dim_x = img.shape

    res = np.zeros((dim_y, dim_x))
    res2 = np.zeros((dim_y, dim_x))
    counter = 1
    for i in range(1, dim_y - 1):
        for j in range(1, dim_x - 1):
            res[i][j] = abs(sum(((img[i-1:i+2, j-1: j+2] * mask).ravel())))
            if res[i][j] > thresh:
                print(f'{counter} : ({i}, {j})')
                res2[i][j] = 255
                counter += 1

    # Save Images after pask and thresholding
    _save('masked_image.jpg', res)
    _save('point_detection.jpg', res2)


def _save_histogram(img):
    """Plots a histogram for the image
    """
    # Removing the 0th value to have a better look at the curve for
    # optimal threshold
    x = np.arange(1, 256)
    img_arr = [[img[i][j] for i in range(img.shape[0])]
               for j in range(img.shape[1])]
    plt.hist(img.ravel()[1:], x)
    plt.xlabel('Values')
    plt.ylabel('Pixel')
    plt.title('Histogram')
    filename = os.path.join(OUTPUT_DIR, 'histogram.jpg')
    plt.savefig(filename)


def _threshold(t, img, show=False):
    """Basic thresholding of an image, given a thresold value
    """
    t_img = np.copy(img)
    for i in range(t_img.shape[0]):
        for j in range(t_img.shape[1]):
            if (t_img[i][j] < t):
                t_img[i][j] = 0
            else:
                t_img[i][j] = 255
    if show:
        cv.imshow('Thresolded Image', t_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        _save('segmented.jpg', t_img)

    return(t_img)


def _segment(img, img_g):
    """Segments the image and returns the boundary
    """
    _save_histogram(img_g)
    t_img = _threshold(204, img_g)

    x = [i for j in range(t_img.shape[1])
         for i in range(t_img.shape[0]) if t_img[i][j] == 255]
    y = [j for j in range(t_img.shape[1])
         for i in range(t_img.shape[0]) if t_img[i][j] == 255]

    max_row, min_row, max_col, min_col = max(
        x), min(x), max(y), min(y)  # Max Row Value
    boundary = [(min_row, max_col), (min_row, min_col),
                (max_row, max_col), (max_row, min_col)]

    return (boundary)


if __name__ == '__main__':

    # Part A : Point detection algoritm

    img1 = cv.imread(img_name_1, 0)
    # Negative Laplacian Filter
    mask = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Threshold the image to get the points, using the mask
    _point_detection(img1, mask, 604)

    # Part B : Point detection algoritm

    img2, img2_g = cv.imread(img_name_2), cv.imread(img_name_2, 0)
    boundary = _segment(img2, img2_g)
    print(f'Segmented Image Bondary: {boundary}')
