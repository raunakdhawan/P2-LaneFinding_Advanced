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
from draw_lane_lines import *
from progressbar import printProgressBar
from sanity_check import *


class Lane:
    def __init__(self):
        # Current lane detections
        self.detected = False
        self.left_poly_pts = []
        self.right_poly_pts = []
        self.left_poly = []
        self.right_poly = []
        self.left_curve_rad = 0.0
        self.right_curve_rad = 0.0
        self.offset = 0.0
        self.detected_lane_img = []
        self.warped_binary = []

        # Will hole previous lane detection
        self.left_poly_pts_previous = []
        self.right_poly_pts_previous = []
        self.left_poly_previous = []
        self.right_poly_previous = []

def find_lanes(raw_image, distortion_coeff, lane):
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
    thresholded_hsl_100 = threshold_hsl(undistorted, 150)  # HSL Thresholding
    _, thresholded_sobel_and_mag = threshold_sobel(undistorted,
                                                   sobel_kernel=7, 
                                                   mag_thresh=(50, 190), 
                                                   grad_thresh=(0.7, 1.2))  # Thresholding with sobel and gradient

    _, hsl_or_mag = threshold_combined(thresholded_hsl_100, thresholded_sobel_and_mag)  # Combine thresholding

    # Apply perspective transform
    src = np.array([[160, 700], [550, 450], [750, 450], [1200, 700]], dtype=np.float32) 
    dst = np.array([[100, 720], [100, 0], [1280, 0], [1280, 720]], dtype=np.float32)
    lane.warped_binary, M, M_inv = perspective_transform(hsl_or_mag, src, dst)

    # Find Lane Pixels if thy are not already detected
    if lane.detected == False:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(lane.warped_binary, 
                                                                 nwindows=10, 
                                                                 margin=100, 
                                                                 minpix=100)
        
        # Fit a polynomial using the found pixels (2nd Order)
        lane.left_poly = np.polyfit(lefty, leftx, 2)
        lane.right_poly = np.polyfit(righty, rightx, 2)
        lane.detected = True
    else:
        leftx, lefty, rightx, righty, out_img = search_around_poly(lane.warped_binary, 
                                                                   lane.left_poly, 
                                                                   lane.right_poly, 
                                                                   50)

        # Fit a polynomial using the found pixels (2nd Order)
        lane.left_poly = np.polyfit(lefty, leftx, 2)
        lane.right_poly = np.polyfit(righty, rightx, 2)

    # Generate x and y values
    ploty = np.linspace(0, lane.warped_binary.shape[0]-1, lane.warped_binary.shape[0])
    left_fitx = lane.left_poly[0]*ploty**2 + lane.left_poly[1]*ploty + lane.left_poly[2]
    right_fitx = lane.right_poly[0]*ploty**2 + lane.right_poly[1]*ploty + lane.right_poly[2]
    lane.left_poly_pts = (left_fitx, ploty)
    lane.right_poly_pts = (right_fitx, ploty)

    # Conversion variables
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    # Find the curvature and position of the vehicle
    y_eval = np.max(ploty)
    lane.left_curve_rad = ((1 + (2*lane.left_poly[0]*y_eval*ym_per_pix + lane.left_poly[1])**2)**1.5)/np.absolute(2*lane.left_poly[0])
    lane.right_curve_rad = ((1 + (2*lane.left_poly[0]*y_eval*ym_per_pix + lane.left_poly[1])**2)**1.5)/np.absolute(2*lane.left_poly[0])

    # Find the offset
    lane.offset = (raw_image.shape[1]//2 - (left_fitx[-1] + right_fitx[-1])/2)*xm_per_pix

    # Sanity check
    print(lane.left_poly)
    lane = sanity_check(raw_image, lane, M_inv)

    return lane

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
    lane = Lane()
    image = cv2.imread(images[4])
    image = cv2.resize(image, (1280, 720))
    lane = find_lanes(image, calib_param, lane)

    # Show the images
    fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(20,10))
    plt1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt2.imshow(cv2.cvtColor(lane.detected_lane_img, cv2.COLOR_BGR2RGB))
    plt.show()

    # On Video
    video_path = "./project_video.mp4"
    # video_path = "./challenge_video.mp4"
    output_video_path = "./output_video_2.avi"

    # Read the video
    video = cv2.VideoCapture(video_path)

    # Video writer
    width = int(video.get(3))
    height = int(video.get(4))
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    out_video = cv2.VideoWriter(output_video_path, 
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                                fps, 
                                (width,height))
    
    # Initialize lane class
    lane = Lane()
    frame_number = 0
    while(video.isOpened()):
        # Read the frame from the video
        ret, frame = video.read()

        frame_number += 1
        # if frame_number/fps < 35 or frame_number/fps > 45:
        #     continue
        if ret == True:
            # Process the frame
            lane = find_lanes(frame, calib_param, lane)

            # # Display the resulting frame
            cv2.imshow('Frame', lane.detected_lane_img)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break   

            # Write to the output file
            out_video.write(lane.detected_lane_img)
            
            # Print output numberf for user
            printProgressBar(frame_number, length, "Progress", length=50)
        
        # Break the loop
        else: 
            print("Cannot read the video")
            break
    
    # When everything done, release the video capture object
    video.release()
    out_video.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
