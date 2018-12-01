import cv2
import numpy as np
import os


def _intensify(img):
    """
    """
    for i, row in enumerate(img):
        for j, ele in enumerate(row):
            if ele >= 0.2:
                img[i][j] = 255
            if ele < 0.2:
                img[i][j] = 0

def _sobel_filter(sobel, img, rad):
    h, w = img.shape
    res = np.asarray([[0.0 for col in range(w)]
                      for row in range(h)])

    for x in range(rad, h-rad):
        for y in range(rad, w-rad):
            _sum = 0
            for i in range(0, (rad*2)+1):
                for j in range(0, (rad*2)+1):
                    _sum += sobel[i][j] * img[x-rad+i][y-rad+j]

            res[x][y] = _sum
    return (res)


def _sobel_edge_detection(img):
    # Sobel Kernels along x and y directions
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # Edge along x directions
    edge_x = _sobel_filter(sobel_x, img, 1)

    # Edge along x directions
    edge_y = _sobel_filter(sobel_y, img, 1)
    return(edge_x, edge_y)


def sobel(img):
    """
    """
    edge_x, edge_y = _sobel_edge_detection(img)

    mag = np.sqrt(edge_x ** 2 + edge_y ** 2)
    mag /= np.max(mag)
    _intensify(mag)
    return (mag)


if __name__ == '__main__':

    img = cv2.imread("./hough.jpg", 0)
    mag = sobel(img)
    # cv2.imshow('Magnitude', mag)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
