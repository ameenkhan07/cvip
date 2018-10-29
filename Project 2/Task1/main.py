import os
import cv2 as cv

OUTPUT_DIR = "outputs/"
img1_name = "./mountain1.jpg"
img2_name = "./mountain2.jpg"


def _save(filename, img):
    """
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # filename = filename+'.png'
    filename = os.path.join(OUTPUT_DIR, filename)
    # print(filename, img.shape)
    cv.imwrite(filename, img)


def _extract_SIFT_keypoints(img_1, img_2):
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

    img1 = cv.imread(img1_name)
    img2 = cv.imread(img2_name)
    print(img1.shape, img2.shape)

    _extract_SIFT_keypoints(img1, img2)

    # gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # sift = cv.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)
    # img = cv.drawKeypoints(gray, kp, img1)
    # # gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # # cv.imwrite('sift_keypoints.jpg', img1)
    # _save('img1', img)
