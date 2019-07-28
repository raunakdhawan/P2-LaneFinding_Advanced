import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import os.path
import pickle
from cal_camera import cal_camera
from perspective_transform import perspective_transform
from threshold import *
from histogram_lane_pixels import *

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
    # Undistort the raw image
    undistorted = cv2.undistort(raw_image, 
                                distortion_coeff["mtx"], 
                                distortion_coeff["dist"], 
                                None, 
                                distortion_coeff["mtx"])

    # Threshold the binary image
    thresholded_hsl_100 = threshold_hsl(undistorted, 100)  # HSL Thresholding
    _, thresholded_sobel_and_mag = threshold_sobel(undistorted,
                                                   sobel_kernel=15, 
                                                   mag_thresh=(50, 190), 
                                                   grad_thresh=(0.7, 1.2))  # Thresholding with sobel and gradient

    _, hsl_or_mag = threshold_combined(thresholded_hsl_100, thresholded_sobel_and_mag)  # Combine thresholding

    # Apply perspective transform
    warped, M, M_inv = perspective_transform(hsl_or_mag)

    # Find lane pixels using the histogram
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped, nwindows=10, margin=100, minpix=100)

    # Fit a polynomial using the found pixels (2nd Order)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    
    return hsl_or_mag, warped

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
        calib_param = {}
        calib_param["mtx"] = mtx
        calib_param["dist"] = dist


    # Load Images
    images = glob.glob("./test_images/*.jpg")

    # Find Lanes
    image = cv2.imread(images[2])
    thresholded, warped = find_lanes(image, calib_param)

    # Show the images
    # fig, (plt1, plt2, plt3) = plt.subplots(1, 3, figsize=(20,10))
    # plt1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt2.imshow(thresholded, cmap="gray")
    # plt3.imshow(warped, cmap="gray")
    # plt.show()

