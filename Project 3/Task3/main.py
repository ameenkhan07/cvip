import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
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


def plot_hough(H):
    """ Plot Hough accumulator matrix
    """
    plot_title = 'Hough Accumulator Plot'
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.show()


def get_hough_transform_lines_acc(img):
    """Create Vote Accumulator array for line using Hough Transformation
    """
    width, height = img.shape

    # Gerenating theta range in radians
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    cos_theta, sin_theta = np.cos(thetas), np.sin(thetas)
    num_thetas = len(thetas)

    # Calculating Rhos for
    diag_len = int(round(np.sqrt(width**2 + height**2)))

    # Initializing Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(
                round(x * cos_theta[t_idx] + y * sin_theta[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, diag_len, thetas, sin_theta, cos_theta


def get_hough_lines(img, accumulator, diag_len, thresh=81, sin=0, cos=1, ang=90, filename='lines.jpg'):
    """Draw line using voted accumulator matrix.
    """
    _img = np.copy(img)
    acc = np.copy(accumulator)

    # Use only thresholded values
    acc = (acc[:, ang]) > thresh
    rhos = np.nonzero(acc)[0]  # Return true indices

    for i in range(0, len(rhos)):
        rho = rhos[i] - diag_len
        _x, _y = rho * cos, rho * sin
        x1 = int(_x + 1000*(-sin))
        y1 = int(_y + 1000*(cos))
        x2 = int(_x - 1000*(-sin))
        y2 = int(_y - 1000*(cos))
        cv.line(_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    _save(filename, _img)


def get_hough_transform_circles_acc(img, diag_len, rad_max, rad_min):
    """Create Vote Accumulator array for circles using Hough Transformation process.
    """
    _img = np.copy(img)
    _im = _img > 0

    # Gerenating theta range, in radians
    theta = np.deg2rad(np.arange(0, 361))
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Intializing 3D Hough Accumulator array of a vs b vs r
    # Range of a and b is caclulated using diag_len
    # Range of radius is approximated from the sourceimage
    acc = np.zeros((2*diag_len, 2*diag_len, rad_max-rad_min), dtype=np.uint8)

    # Vote in the hough accumulator
    _y, _x = np.nonzero(_im)
    for i in range(0, len(_x)):  # For all edges
        x, y = _x[i], _y[i]
        # Accumulator for all radius in the range provided
        for rad in range(rad_min, rad_max):
            for t_ind in range(0, 361):
                a = int(round(x - (rad * cos_theta[t_ind]))) + diag_len
                b = int(round(y - (rad * sin_theta[t_ind]))) + diag_len
                # print(f'{a},{b}')
                acc[a, b, rad-rad_min] += 1
    return(acc)


def get_hough_circles(img, acc, diag_len, rad_min, thresh, filename='temp.jpg'):
    """Draw circles using voted accumulator matrix
    """
    _img = np.copy(img)

    # Thresholding accumulator values
    acc = acc > thresh
    acc = np.nonzero(acc)

    _A, _B, rad = acc[0], acc[1], acc[2]

    for i in range(0, len(_A)):
        a, b, r = _A[i] - diag_len, _B[i] - diag_len, rad[i] + rad_min
        cv.circle(_img, (a, b), r, (0, 255, 0), 1)

    _save(filename, _img)


if __name__ == '__main__':

    img, img_g = cv.imread(img_name), cv.imread(img_name, 0)

    # Implement Canny Edge Detection Algo ?
    edges = sobel(img_g)

    # Accumulator Voting Matrix for Lines
    acc_mat, diag_len, thetas, sin_t, cos_t = get_hough_transform_lines_acc(
        edges)
    # plot_hough(acc_mat)

    # Part A : Detecting Vertical Lines
    get_hough_lines(img, acc_mat, diag_len, thresh=81, sin=0,
                    cos=1, ang=90, filename='red_line.jpg')
    print('Detected Vertical Lines drawn')

    # Part B : Detecting Diagonal Lines
    _cos, _sin = cos_t[55], sin_t[55]
    get_hough_lines(img, acc_mat, diag_len, thresh=120,
                    sin=_sin, cos=_cos, ang=55, filename='blue_line.jpg')
    print('Detected Diagonal Lines drawn')


    # Part C : Detecting Circles

    rad_min, rad_max = 20, 30  # Range of radius of the coins, assumed
    # Accumulator Voting Matrix for Circles
    circle_acc_mat = get_hough_transform_circles_acc(
        edges, diag_len, rad_max, rad_min)
    # plot_hough(circle_acc_mat)

    get_hough_circles(img, circle_acc_mat, diag_len, rad_min,
                      181, filename='coin.jpg')
    print('Detected Circles drawn')
