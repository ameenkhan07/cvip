import cv2
import numpy as np
import os

OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def _pad(img, pad):
    """Returns the padded image
    """
    dim_y, dim_x = img.shape
    pad_y, pad_x = pad
    padded_img = np.asarray([[0 for i in range(dim_x + (pad_x * 2))]
                             for j in range(dim_y + (pad_y * 2))],
                            dtype=np.float32)
    padded_img[pad_x:-pad_x, pad_y:-pad_y] = img
    return padded_img


def _convolve(img, kernel_x, kernel_y, dim_x, dim_y):
    """Covolve img with sobel operator along x and y simultaneously
    """
    # Output buffer array for storing results
    output_x = np.asarray([[0 for i in range(dim_x)] for j in range(dim_y)],
                          dtype=np.uint8)
    output_y = np.asarray([[0 for i in range(dim_x)] for j in range(dim_y)],
                          dtype=np.uint8)

    # Sobel Filter (Convolution of image with Kernel)
    for x in range(1, dim_y - 1):
        for y in range(1, dim_x - 1):
            #note: parts of the image multiplied by the 0 portions of the filters
            temp_x = abs(
                sum(map(sum, img[x - 1:x + 2, y - 1:y + 2] * kernel_x)))
            if temp_x > 255: temp_x = 255
            output_x[x - 1, y - 1] = temp_x

            temp_y = abs(
                sum(map(sum, img[x - 1:x + 2, y - 1:y + 2] * kernel_y)))
            if temp_y > 255: temp_y = 255
            output_y[x - 1, y - 1] = temp_y

    return (output_x, output_y)


if __name__ == '__main__':
    ## Setting up variables
    img = cv2.imread("./task1.png", 0)
    dim_y, dim_x = img.shape

    # Add 0 - Padding (Not to lose information for edges)
    img = _pad(img, (1, 1))

    # Kernel Filters (Flipped for convolution)
    kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    output_x, output_y = _convolve(img, kernel_x, kernel_y, dim_x, dim_y)

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'Sobel_x.png'), output_x)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'Sobel_y.png'), output_y)

    # cv2.imshow('Sobel Output image along x', np.asarray(output_x))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow('Sobel Output image along x', np.asarray(output_y))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()