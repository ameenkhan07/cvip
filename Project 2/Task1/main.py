import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random

UBIT = 'ameenmoh'
np.random.seed(sum([ord(c) for c in UBIT]))
random.seed(sum([ord(c) for c in UBIT]))

OUTPUT_DIR = "outputs/"
img1_name = "./mountain1.jpg"
img2_name = "./mountain2.jpg"


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
    _save('task1sift1.jpg', img_keypoint)

    # Get keypoints and descriptor for image 2
    keypoint_2, descriptor_2 = sift_obj.detectAndCompute(img2_g, None)
    # Save Keypoint Image for image 2
    img_keypoint = cv.drawKeypoints(img2_g, keypoint_2, img1)
    _save('task1sift2.jpg', img_keypoint)

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
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1_g, keypoint_1, img2_g,
                             keypoint_2, good_matches, None, flags=2)
    _save('task1_matches_knn.jpg', img3)

    return(good_matches)


def _get_homography_matrix(good_matches, keypoint_1, keypoint_2, _H=True):
    """Homography matrix H (with RANSAC) from the image 1 to image 2
    """
    source = np.float32(
        [keypoint_1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dest = np.float32(
        [keypoint_2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Mask contains both inliers and outliers
    H, mask = cv.findHomography(source, dest, cv.RANSAC, 5.0)
    if _H:
        print("Homography Matrix  : ")
        print(H)
    return(mask, H)


def _draw_inlier_match_keypoints(img1_g, keypoint_1, img2_g, keypoint_2, good_matches, mask):
    """Draw match image for  10 random matches using only inliers.
    """
    matchesMask = mask.ravel().tolist()
    # Inlier points are ones where mask is 1
    inlier_index = random.sample(
        [i for i, ele in enumerate(matchesMask) if ele == 1], 10)
    matchesMask = [matchesMask[i] for i in inlier_index]
    good_matches = [good_matches[i] for i in inlier_index]
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # print(len(matchesMask))
    # Get the elements inside of list elements
    good_matches = [i[0] for i in good_matches]
    img3 = cv.drawMatches(img1_g, keypoint_1, img2_g,
                          keypoint_2, good_matches, None, **draw_params)
    _save('task1_matches.jpg', img3)


def _draw_stitched_image(img1, img2, H_Matrix):
    """Warp and stitch 2 images using Homography matrix as perspective reference
    """

    # Get width and height of input images
    (w1, h1), (w2, h2) = img1.shape, img2.shape
    # Get the canvas dimesions
    img1_dims = np.float32(
        [[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    # Get relative perspective of second image with first image
    img2_dims = cv.perspectiveTransform(np.float32(
        [[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2), H_Matrix)
    # Append Dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Affine transformation
    transformation = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]])

    # Warp images to get the resulting image
    res = cv.warpPerspective(img2, transformation.dot(H_Matrix),
                             (x_max-x_min, y_max-y_min))
    res[-y_min:w1+-y_min, -x_min:h1+-x_min] = img1
    _save('task1_pano.jpg', res)


if __name__ == '__main__':

    img1, img1_g = cv.imread(img1_name), cv.imread(img1_name, 0)
    img2, img2_g = cv.imread(img2_name), cv.imread(img2_name, 0)

    sift_obj = cv.xfeatures2d.SIFT_create()

    # Part 1
    keypoint_1, descriptor_1, keypoint_2, descriptor_2 = _get_SIFT_keypoints(
        sift_obj, img1, img2, img1_g, img2_g)

    # Part 2
    good_matches = _draw_match_keypoints(sift_obj, img1_g, keypoint_1, descriptor_1,
                                         img2_g, keypoint_2, descriptor_2)

    # Part 3
    mask, H_Matrix = _get_homography_matrix(
        good_matches, keypoint_1, keypoint_2)

    # Part 4
    _draw_inlier_match_keypoints(
        img1_g, keypoint_1, img2_g, keypoint_2, good_matches, mask)

    # Part 5
    _draw_stitched_image(img2_g, img1_g, H_Matrix)
