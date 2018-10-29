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

    """
    """
    sift_create_obj = cv.xfeatures2d.SIFT_create()

    # Create keypoint for image 1
    img_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    keypoint_obj = sift_create_obj.detect(
        cv.cvtColor(img_1, cv.COLOR_BGR2GRAY), None)
    img_keypoint_1 = cv.drawKeypoints(img_gray, keypoint_obj, img_1)
    _save('task1sift1.jpg', img_keypoint_1)

    # Create keypoint for image 2
    img_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    keypoint_obj = sift_create_obj.detect(
        cv.cvtColor(img_2, cv.COLOR_BGR2GRAY), None)
    img_keypoint_2 = cv.drawKeypoints(img_gray, keypoint_obj, img_2)
    _save('task1sift2.jpg', img_keypoint_2)

    return(img_keypoint_1, img_keypoint_2)


if __name__ == '__main__':

    img1, img1_g = cv.imread(img1_name), cv.imread(img1_name, 0)
    img2, img2_g = cv.imread(img2_name), cv.imread(img2_name, 0)

    sift_obj = cv.xfeatures2d.SIFT_create()

    keypoint_1, descriptor_1, keypoint_2, descriptor_2 = _extract_SIFT_keypoints(
        sift_obj, img1, img2, img1_g, img2_g)

