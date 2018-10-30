import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random

UBIT = 'ameenmoh'
np.random.seed(sum([ord(c) for c in UBIT]))

OUTPUT_DIR = "outputs/"
img_left_name = "./tsucuba_left.png"
img_right_name = "./tsucuba_right.png"


def _save(filename, img):
    """Saves the image with filename in output dir 
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # filename = filename+'.png'
    filename = os.path.join(OUTPUT_DIR, filename)
    # print(filename, img.shape)
    cv.imwrite(filename, img)


def _get_SIFT_keypoints(sift_obj, img1, img2, img1_g, img2_g):
    """Extract SIFT features(keypoints and descriptors) and 
    draw the keypoints and return the images
    """
    # Get keypoints and descriptor for image 1
    keypoint_1, descriptor_1 = sift_obj.detectAndCompute(img1_g, None)
    # Save Keypoint Image for image 1
    img_keypoint = cv.drawKeypoints(img1_g, keypoint_1, img1)
    _save('task2sift1.jpg', img_keypoint)

    # Get keypoints and descriptor for image 2
    keypoint_2, descriptor_2 = sift_obj.detectAndCompute(img2_g, None)
    # Save Keypoint Image for image 2
    img_keypoint = cv.drawKeypoints(img2_g, keypoint_2, img1)
    _save('task2sift2.jpg', img_keypoint)

    return(keypoint_1, descriptor_1, keypoint_2, descriptor_2)


def _draw_match_keypoints(*args):
    """Draw match image for all matches
    Keypoint matching using k-nearest neighbour
    """
    sift_obj = args[0]
    img1_g, img2_g = args[1], args[4]
    keypoint_1, descriptor_1 = args[2], args[3]
    keypoint_2, descriptor_2 = args[5], args[6]

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    # All matches, inliers and outliers
    # good = [[i[0]] for i in matches]
    # print(len(good_matches), len(good))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1_g, keypoint_1, img2_g,
                             keypoint_2, good_matches, None, flags=2)
    _save('task2_matches_knn.jpg', img3)

    return(good_matches)


def _get_fundamental_matrix(good_matches, keypoint_1, keypoint_2, F_=True):
    """
    """
    source = np.float32(
        [keypoint_1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dest = np.float32(
        [keypoint_2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    F, mask = cv.findFundamentalMat(source, dest)

    if F_:
        print('Fundamental Matrix')
        print(F)
    return F, mask, source, dest


if __name__ == '__main__':

    img1, img1_g = cv.imread(img_left_name), cv.imread(img_left_name, 0)
    img2, img2_g = cv.imread(img_right_name), cv.imread(img_right_name, 0)

    sift_obj = cv.xfeatures2d.SIFT_create()

    # Part 1
    keypoint_1, descriptor_1, keypoint_2, descriptor_2 = _get_SIFT_keypoints(
        sift_obj, img1, img2, img1_g, img2_g)

    good_matches = _draw_match_keypoints(
        sift_obj, img1_g, keypoint_1, descriptor_1, img2_g, keypoint_2, descriptor_2)

    # Part 3
    F_Matrix, inliers, source, dest = _get_fundamental_matrix(
        good_matches, keypoint_1, keypoint_2)
