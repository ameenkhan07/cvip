import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
# import math
from sobel import sobel

OUTPUT_DIR = "outputs/"
img_name = "./hough.jpg"


def _save(filename, img):
    """Saves the image with filename in output dir 
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filename = os.path.join(OUTPUT_DIR, filename)
    cv.imwrite(filename, img)


def get_hough_transform_acc(img):
    """
    """
    width, height = img.shape

    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    num_thetas = len(thetas)

    # Calculating Rhos for
    diag_len = int(round(np.sqrt(width**2 + height**2)))

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(
                round(x * cos_theta[t_idx] + y * sin_theta[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, diag_len, thetas, sin_theta, cos_theta


def plot_hough(H):
    """ Plot Hough accumulator matrix
    """
    plot_title = 'Hough Accumulator Plot'
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.show()


def get_hough_lines(img, accumulator, diag_len, thresh=81, sin=0, cos=1, ang=90, filename='lines.jpg'):
    """
    """
    _img = np.copy(img)
    acc = np.copy(accumulator)

    acc = (acc[:, ang]) > thresh
    rhos = np.nonzero(acc)[0]  # Return true indices

    for i in range(0, len(rhos)):
        rho = rhos[i]
        rho = rho - diag_len
        x0 = rho * cos
        y0 = rho * sin
        x1 = int(x0 + 1000*(-sin))
        y1 = int(y0 + 1000*(cos))
        x2 = int(x0 - 1000*(-sin))
        y2 = int(y0 - 1000*(cos))
        cv.line(_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    _save(filename, _img)


if __name__ == '__main__':

    img, img_g = cv.imread(img_name), cv.imread(img_name, 0)

    # Implement Canny Edge Detection Algo ?
    edges = sobel(img_g)

    acc_mat, diag_len, thetas, sin_t, cos_t = get_hough_transform_acc(edges)

    # Detecting Vertical Lines
    get_hough_lines(img, acc_mat, diag_len, thresh=81, sin=0,
                    cos=1, ang=90, filename='red_line.jpg')

    # Detecting Angled Lines
    get_hough_lines(img, acc_mat, diag_len, thresh=120, sin=sin_t[55], cos=cos_t[55], ang = 55, filename = 'blue_line.jpg')
