import math
import numpy as np
import cv2


class Gaussian:
    def __init__(self, sig):
        self.sigma = sig

    def _gaussian_func(self, x, y):
        """Gaussian Function used in defining kernel
        """

        num = -(x**2 + y**2)
        e = np.exp(num / (2 * self.sigma**2))
        return (1 / ((2 * math.pi) * self.sigma**2)) * e

    def _normalize_kernel(self, kernel):
        """Normalizes kernel matrice with a factor of C=sum(kernel)
        """
        c = sum(map(sum, kernel))
        kernel = [[(1 / c) * i for i in j] for j in kernel]
        return kernel

    def _gaussian_kernel(self):
        """Creates and returns a 7*7 2D Gaussian Kernel
        """
        kernel = [[self._gaussian_func(x, y) for x in range(-3, 4)]
                  for y in range(3, -4, -1)]
        kernel = self._normalize_kernel(kernel)
        return np.asarray(kernel)

    def _flip(self, kernel):
        """Returns horizontally and vertically flipped matrices
        """
        # Horizontally flipped (reverse)
        kernel = kernel[:, ::-1]
        # Vertically flipped (reverse)
        kernel = kernel[::-1, ...]
        return kernel

    def _pad(self, img, pad):
        """Returns the padded image
        """
        dim_y, dim_x = img.shape
        pad_y, pad_x = pad
        padded_img = np.asarray([[0 for i in range(dim_x + (pad_x * 2))]
                                 for j in range(dim_y + (pad_y * 2))],
                                dtype=np.float32)
        padded_img[pad_x:-pad_x, pad_y:-pad_y] = img
        return padded_img

    def _gaussian_filter(self, img):
        """Returns Gaussian Filtered image
        """
        kernel = self._gaussian_kernel()
        kernel = self._flip(kernel)
        # print(kernel.shape)
        # print(img.shape)
        dim_y, dim_x = img.shape

        ## Pad the image before convolving
        # img = np.pad(img, (3,3), 'edge')
        img = self._pad(img, (3, 3))

        ## Pad the output Image
        output = np.asarray([[0 for i in range(dim_x)] for j in range(dim_y)],
                            dtype=np.uint8)

        for x in range(3, dim_y + 3):
            for y in range(3, dim_x + 3):
                temp_x = abs(
                    sum(map(sum, img[x - 3:x + 4, y - 3:y + 4] * kernel)))
                output[x - 3, y - 3] = temp_x
        return output


if __name__ == '__main__':
    # For Testing gaussian kernel
    img = cv2.imread("./task2.jpg", 0)
    g = Gaussian(1 / math.sqrt(2))
    img = g._gaussian_filter(img)

    cv2.imshow("name", np.asarray(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
