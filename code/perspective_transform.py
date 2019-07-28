import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cal_camera import cal_camera


def perspective_transform(undistorted, src=np.float32([[100, 670], [620, 440], [730, 440], [1100, 670]]), 
                                       dst=np.float32([[100, 720], [100, 0], [1280, 0], [1280, 720]])):
    
    # Convert to gray scale
    if len(undistorted.shape) > 2:
        undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    else:
        undistorted_gray = undistorted    
    img_size = undistorted.shape

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Get the inverse perspective transformation matrix
    M_inv = cv2.getPerspectiveTransform(dst, src) 

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undistorted_gray, M, (img_size[1], img_size[0]))

    return warped, M, M_inv


if __name__ == "__main__":
    # Read in the image
    test_img_path = "./test_images/test5.jpg"
    test_img_path = "./test_images/straight_lines1.jpg"
    test_img = cv2.imread(test_img_path)

    # Undistort the image
    pickle_file_path = "./code/calibration.p"

    # Load the calibration parameters from pickled file else calculate again
    with open(pickle_file_path, "rb") as pickle_file:
        calib_param = pickle.load(pickle_file)

    undistorted = cv2.undistort(test_img, 
                                calib_param["mtx"], 
                                calib_param["dist"], 
                                None, 
                                calib_param["mtx"])

    warped, M, M_inv = perspective_transform(undistorted)
    inversed = cv2.warpPerspective(warped, M_inv, (warped.shape[1], warped.shape[0]))

    fig, (plt1, plt2, plt3) = plt.subplots(1, 3, figsize=(20,10))
    plt1.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt2.imshow(warped, cmap="gray")
    plt3.imshow(inversed, cmap="gray")
    plt.show()    