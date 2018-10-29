import os
import cv2 as cv

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


def _extract_SIFT_keypoints(sift_obj, img1, img2, img1_g, img2_g):
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


def _match_keypoints(*args):
    """
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
        if m.distance < 0.75*n.distance: good_matches.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1_g, keypoint_1, img2_g,
                             keypoint_2, good_matches, None, flags=2)
    _save('task1 matches knn.jpg', img3)


if __name__ == '__main__':

    img1, img1_g = cv.imread(img1_name), cv.imread(img1_name, 0)
    img2, img2_g = cv.imread(img2_name), cv.imread(img2_name, 0)

    sift_obj = cv.xfeatures2d.SIFT_create()

    keypoint_1, descriptor_1, keypoint_2, descriptor_2 = _extract_SIFT_keypoints(
        sift_obj, img1, img2, img1_g, img2_g)

