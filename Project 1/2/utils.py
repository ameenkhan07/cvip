
# Gaussian Filter
def _gaussian_func(x, y, sig):
    """Gaussian Function used in defining kernel
    """
    pi = 3.14
    num = -(x**2 + y**2)
    e = np.exp( num / ( 2.0 * sig**2 ) )
    return (1 /( (2*pi) * sig**2)) * e

def _gaussian_kernel(sig):
    """
    Creates a 7*7 2D Gaussian Kernel
    """
    # print(_gaussian_func(-2, 2, 1/math.sqrt(2)))
    kernel = [[0 for x in range(7)] for y in range(7)]
    for x in range(-3,3):
        for y in range(-3,3):
            kernel[x][y] = _gaussian_func(x, y, sig)
    print(kernel)
    return kernel

def _flip(mat):
    """Retruns horizontally and vertically flipped Matrices
    """
    # Horizontally flipped (reverse)
    mat =  mat[:, ::-1]
    # Vertically flipped (reverse)
    mat = mat[::-1, ...]
    return mat
    
def _conv(img, kernel):
    """
    """
    dim_y, dim_x = img.shape

    # Flip kernel before convolution
    # img = s_flip(img)
    print(".........")
    output = np.zeros((dim_y,dim_x), np.uint8)
    print(kernel.shape, img.shape)
    # Convolve img and kernel
    for x in range(1, dim_y-1):
        for y in range(1, dim_x-1):
            temp_x = abs(sum(map(sum, img[x-1:x+6, y-1:y+6] * kernel)))
            if temp_x>255: temp_x=255
            output[x-1, y-1] = temp_x
    return output
    # cv2.imshow('Gaussian Images',np.asarray(output))
