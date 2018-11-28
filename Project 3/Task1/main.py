import cv2 as cv
import numpy as np
import os

OUTPUT_DIR = "outputs/"
img_name = "./noise.jpg"


class MorphImageProcessing:

    def __init__(self, img, str_img):
        self.img = img
        self.str_img = str_img

    @staticmethod
    def _save(filename, img):
        """Saves the image with filename in output dir 
        """
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        filename = os.path.join(OUTPUT_DIR, filename)
        cv.imwrite(filename, img)

    def _dilate(self, img, str_img=[]):
        '''Expands the poindary of img
        '''
        w, h = img.shape
        res = np.asarray([[0 for _ in range(h)] for _ in range(w)])
        # Iterating the anchor of the structure image over the original image
        for i, im_row in enumerate(img):
            for j, im_ele in enumerate(im_row):
                if img[i][j]:
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (i+k) >= w or (i+k) < 0 or (l+j) >= h or (l+j) < 0:
                                continue
                            else:
                                res[i+k][j+l] = 255

        return(res)

    def _erode(self, img, str_img=[]):
        '''Contracts the boundary of img
        '''
        w, h = img.shape
        se_w, se_h = str_img.shape
        res = np.asarray([[0 for _ in range(h)] for _ in range(w)])
        for i, im_row in enumerate(img):
            for j, im_ele in enumerate(im_row):
                flag = True
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if (i+k) >= w or (i+k) < 0 or (l+j) >= h or (l+j) < 0:
                            continue
                        if str_img[k][l] != 0 and img[i+k][j+l] == 0:
                            flag = False
                if flag:
                    res[i][j] = 255

        return(np.asarray(res))

    def opening(self, img=[], str_img=[]):
        """Morph Composite operation, first erode then dilate
        """
        if not len(img):
            img = self.img
        temp = self._erode(img, self.str_img)
        res = self._dilate(temp, str_img)
        return res

    def closing(self, img=[], str_img=[]):
        """Morph Composite operation, first dilute then erode
        """
        if not len(img):
            img = self.img
        temp = self._dilate(img, self.str_img)
        res = self._erode(temp, self.str_img)
        return res


if __name__ == '__main__':
    img = cv.imread(img_name, 0)

    str_img = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    Morph = MorphImageProcessing(img, str_img)
    # Denoising
    # Approach 1 : Opening then Closing
    print(f'\nApproach 1 : Opening followed by Closing morph operations')
    temp1 = Morph.opening(img, str_img)
    res1 = Morph.closing(temp1, str_img)
    Morph._save('res_noise1.png', res1)
    print('\nres_noise1.png saved!!')

    # Approach 2 : Closing then Opening
    print(f'\nApproach 2 : Closing followed by Opening morph operations')
    temp2 = Morph.closing(img, str_img)
    res2 = Morph.opening(temp2, str_img)
    Morph._save('res_noise2.png', res2)
    print('\nres_noise2.png saved!!')

    # Boundary Extraction

    print(f'\nBoundary of Approach 1')
    boundary1 = np.subtract(res1, Morph._erode(res1, str_img))
    Morph._save('res_bound1.png', boundary1)

    print(f'\nBoundary of Approach 2')
    boundary2 = np.subtract(res2, Morph._erode(res2, str_img))
    Morph._save('res_bound2.png', boundary2)
