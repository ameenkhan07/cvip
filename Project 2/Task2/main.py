import os
import cv2 as cv
import numpy as np
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


def _get_fundamental_matrix(good_matches, keypoint_1, keypoint_2, F_ = True):
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


def drawlines(img1, img2, lines, pts1, pts2):
    """ img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines 
    """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        # img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        # img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return(img1, img2)


def _draw_inlier_matches(img1_g, img2_g, good_matches, keypoint_1, keypoint_2):
    """
    """
    good_matches = random.sample(good_matches, 10)
    F, inliers, pts1, pts2 = _get_fundamental_matrix(
        good_matches, keypoint_1, keypoint_2, False)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_g, img2_g, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2_g, img1_g, lines2, pts2, pts1)

    # plt.subplot(121), plt.imshow(img5)
    _save('task2_epi_right.jpg', img5)
    # plt.subplot(122), plt.imshow(img3)
    _save('task2_epi_left.jpg', img3)


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

    _draw_inlier_matches(img1_g, img2_g, good_matches, keypoint_1, keypoint_2)
