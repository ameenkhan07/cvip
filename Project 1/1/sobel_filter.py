import cv2
import numpy as np

## Setting up variables
img = cv2.imread("./task1.png", 0)
dim_y, dim_x = img.shape 

# Add 0 - Padding (Not to lose information for edges)

img = np.pad(img, (1,1), 'edge')


# Kernel Filters (Flipped for convolution)
kernel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
kernel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]

# Output
output_x = np.zeros((dim_y,dim_x))
output_y = np.zeros((dim_y,dim_x))


# Sobel Filter (Convolution of image with Kernel)
for x in range(1, dim_y-1):
    for y in range(1, dim_x-1):
        #note: parts of the image multiplied by the 0 portions of the filters
        tempx = sum(map(sum, img[x-1:x+2, y-1:y+2] * kernel_x))
        # if tempx>255: tempx=255
        output_x[x-1, y-1] = tempx
        output_y[x-1, y-1] = sum(map(sum, img[x-1:x+2, y-1:y+2] * kernel_y))

cv2.imshow('Sobel Output image along x',np.asarray(output_x))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Sobel Output image along x',np.asarray(output_y))
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Check if True
# edge_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
# print(edge_x.size, np.asarray(output_x).size)
# print(np.array_equal(edge_x, np.asarray(output_x)))
