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


def find_lanes(raw_image, distortion_coeff, lane_left, lane_right):
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
    src = np.array([[160, 700], [550, 450], [750, 450], [1200, 700]], dtype=np.float32) 
    dst = np.array([[100, 720], [100, 0], [1280, 0], [1280, 720]], dtype=np.float32)
    warped, M, M_inv = perspective_transform(hsl_or_mag, src, dst)

    # Find Lane Pixels if thy are not already detected
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped, nwindows=10, margin=100, minpix=100) 

    # Fit a polynomial using the found pixels (2nd Order)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Fill the values in the lane_left and lane_right class
    lane_left.current_fit = left_fit
    lane_right.current_fit = right_fit
    lane_left.detected = True
    lane_right.detected = True

    # Conversion variables
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    # Find the curvature and position of the vehicle
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])

    # Find the offset
    offset = (raw_image.shape[1]//2 - (left_fitx[-1] + right_fitx[-1])/2)*xm_per_pix
    

    # Draw the lane lines on the raw image
    with_lanes = color_lane(raw_image, 
                            warped, 
                            (left_fitx, ploty), 
                            (right_fitx, ploty), 
                            (leftx, lefty), 
                            (rightx, righty), 
                            M_inv)

    # Write text on the image
    with_lanes_text = with_lanes.copy()
    cv2.putText(with_lanes_text, 
                "Curvature (Left Lane): {:.2f} m".format(left_curverad), 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255,255,255), 
                2)
    cv2.putText(with_lanes_text, 
                "Curvature (Right Lane): {:.2f} m".format(right_curverad),
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1., 
                (255,255,255), 
                2)
    cv2.putText(with_lanes_text, 
                    "Car Offset : {:.2f} m".format(offset),
                    (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1., 
                    (255,255,255), 
                    2)

    
    return hsl_or_mag, warped, with_lanes_text

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

    # Initialize the left and right lane
    left_lane = Lane()
    right_lane = Lane()

    # Find Lanes
    image = cv2.imread(images[1])
    thresholded, warped, with_lanes = find_lanes(image, calib_param, left_lane, right_lane)

    # Show the images
    # fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2, figsize=(20,10))
    # plt1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt2.imshow(thresholded, cmap="gray")
    # plt3.imshow(warped, cmap="gray")
    # plt4.imshow(cv2.cvtColor(with_lanes, cv2.COLOR_BGR2RGB))
    # plt.show()

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
    frame_number = 0
    while(video.isOpened()):
        # Read the frame from the video
        ret, frame = video.read()
        
        if ret == True:
            # Process the frame
            thresholded, warped, with_lanes = find_lanes(frame, calib_param, left_lane, right_lane)
            
            # Display the resulting frame
            cv2.imshow('Frame', with_lanes)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break   

            # Write to the output file
            out_video.write(with_lanes)
            
            # Print output numberf for user
            frame_number += 1
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
