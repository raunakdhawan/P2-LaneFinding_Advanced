import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os.path
import pickle
from cal_camera import cal_camera

def find_lanes(raw_image, distortion_coeff):
    '''
    This is the main function that will take the raw image as input and find the lane lines using the following pipeline
    1. Use the distortion coefficients and apply distortion correction
    2. Threshold the image to get a binary image
    3. Apply perspective transform to get the birds eye view
    4. Detect the lane pixels
    5. Calculate the curvature and find the location of the vehicle w.r.t center
    6. Warp the lanes found back to the original image
    7. Return an image with the lanes found
    '''
    return raw_image

if __name__ == "__main__":
    # Calibration of Camera
    pickle_file_path = "./code/calibration.p"

    # Load the calibration parameters from pickled file else calculate again
    if os.path.isfile(pickle_file_path):
        with open(pickle_file_path, "rb") as pickle_file:
            calib_param = pickle.load(pickle_file)
    else:
        # Calibration of Camera
        path_imgs = "./camera_cal/calibration*.jpg"
        num_x = 9
        num_y = 5

        # Calculate calibration parameters
        ret, mtx, dist, rvecs, tvecs = cal_camera(path_imgs, num_x, num_y)


    # Load Images
    images = glob.glob("./test_images/*.jpg")


