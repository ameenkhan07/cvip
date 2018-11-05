import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

UBIT = 'ameenmoh'
np.random.seed(sum([ord(c) for c in UBIT]))
random.seed(sum([ord(c) for c in UBIT]))

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


def drawlines(img1, img2, lines, pts1, pts2, colorArr):
    """
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    color_counter = 0  # For color counter
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = colorArr[color_counter]
        color_counter += 1
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
    return img1, img2


def _draw_inlier_matches(img1_g, img2_g, mask, fundamental, src_pts, dst_pts):
    """
    """
    # Inlier points filtering from given set of src/dest points
    dest_pts, src_pts = dst_pts[mask.ravel() == 1], src_pts[mask.ravel() == 1]
    # print(dest_pts.shape, src_pts.shape)

    # Select random 10 points in the inlier points
    r_ = np.random.randint(0, src_pts.shape[0], 10)

    dest_pts = np.asarray([dest_pts[i] for i in r_])
    src_pts = np.asarray([src_pts[i] for i in r_])

    lines1 = cv.computeCorrespondEpilines(
        src_pts.reshape(-1, 1, 2), 2, fundamental).reshape(-1, 3)

    # Select Random 10 colors for the epilines in left and right images
    random_col_list = [tuple(np.random.randint(
        0, 255, 3).tolist()) for _ in dest_pts]

    img5, img6 = drawlines(img1_g, img2_g, lines1,
                           dest_pts, src_pts, random_col_list)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(
        dest_pts.reshape(-1, 1, 2), 1, fundamental).reshape(-1, 3)
    img3, img4 = drawlines(img2_g, img1_g, lines2,
                           src_pts, dest_pts, random_col_list)
    _save('task2_epi_right.jpg', img3)
    _save('task2_epi_left.jpg', img3)

def _draw_disparity_map(img1, img2):
    """
    """
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=16,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )
    disparity = stereo.compute(img1, img2).astype(np.float32) / 8.0
    # disparity = (disparity - min_disp)/num_disp
    # plt.imshow(disparity,'task2 disparity')
    # _save_plot('task2_disparity.jpg', np.asarray(disparity))
    _save('task2_disparity.jpg', disparity)
    # plt.show()


if __name__ == '__main__':

    img1, img1_g = cv.imread(img_left_name), cv.imread(img_left_name, 0)
    img2, img2_g = cv.imread(img_right_name), cv.imread(img_right_name, 0)

    sift_obj = cv.xfeatures2d.SIFT_create()

    # Part 1
    keypoint_1, descriptor_1, keypoint_2, descriptor_2 = _get_SIFT_keypoints(
        sift_obj, img1, img2, img1_g, img2_g)

    good_matches = _draw_match_keypoints(
        sift_obj, img1_g, keypoint_1, descriptor_1, img2_g, keypoint_2, descriptor_2)

    # Part 2
    F_Matrix, mask, source, dest = _get_fundamental_matrix(
        good_matches, keypoint_1, keypoint_2)

    # Part 3 : Inlier Epilines for the 2 images
    _draw_inlier_matches(img1_g, img2_g, mask, F_Matrix, source, dest)

    # Part 4 : Draw Disparity Map
    _draw_disparity_map(img1_g, img2_g)
