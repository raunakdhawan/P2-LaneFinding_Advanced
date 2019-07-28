import cv2
import numpy as np
from matplotlib import pyplot as plt


def threshold_hsl(raw_image, thresh_min=100):
    '''
    Options for thresholding
    1. Use HSL
    2. Use the Sobel with gradient
    3. Use histogram
    4. Combine the above to create one thresholded image
    '''
    # Gaussian Blur
    # blurred = cv2.GaussianBlur(raw_image, (3, 3), 0)
    blurred = raw_image
    
    # Convert to HSL
    img_hsl = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    img_h, img_s, img_l = img_hsl[:, :, 0], img_hsl[:, :, 2], img_hsl[:, :, 1]

    # A blank canvas
    hsl_threshold = np.zeros(img_s.shape, dtype=np.int32)

    # Threshold
    hsl_threshold[img_s > thresh_min] = 1

    # Make the upper half black
    hsl_threshold[:int(hsl_threshold.shape[0]/2), :] = 0

    return hsl_threshold

def threshold_sobel(img, sobel_kernel=3, mag_thresh=(0, 255), grad_thresh=(0, np.pi/2)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the x and y gradients
    abs_x = np.abs(sobelx)
    abs_y = np.abs(sobely)

    # Calculate scaled absloute
    abs_scaled_x = np.uint8(255*abs_x/np.max(abs_x))
    abs_scaled_y = np.uint8(255*abs_y/np.max(abs_y))

    # Binary Output X abs scaled
    binary_out_x_scaled = np.zeros_like(abs_scaled_x)
    binary_out_x_scaled[(abs_scaled_x >= 20) & (abs_scaled_x <= 100)] = 1

    # Binary Output y abs scaled
    binary_out_y_scaled = np.zeros_like(abs_scaled_y)
    binary_out_y_scaled[(abs_scaled_y >= 20) & (abs_scaled_y <= 100)] = 1

    # Calculate sobel magnitude and scale it
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))

    # Binary output
    binary_output_mag = np.zeros_like(scaled_sobel)
    binary_output_mag[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    atan = np.arctan2(abs_y, abs_x)
    
    # Create a binary mask where direction thresholds are met
    binary_output_grad = np.zeros_like(atan)
    binary_output_grad[(atan >= grad_thresh[0]) & (atan <= grad_thresh[1])] = 1

    # Combine the binary outputs (Using x and grad)
    binary_and_x_grad = np.zeros_like(binary_output_grad)
    binary_and_x_grad[(binary_out_x_scaled == 1) & (binary_output_grad == 1)] = 1
    
    binary_or_x_grad = np.zeros_like(binary_output_grad)
    binary_or_x_grad[(binary_out_x_scaled == 1) | (binary_output_grad == 1)] = 1

    # Combine the binary outputs (Using mag and grad)
    binary_and_mag_grad = np.zeros_like(binary_output_grad)
    binary_and_mag_grad[(binary_output_mag == 1) & (binary_output_grad == 1)] = 1
    
    binary_or_mag_grad = np.zeros_like(binary_output_grad)
    binary_or_mag_grad[(binary_output_mag == 1) | (binary_output_grad == 1)] = 1

    # fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2)
    # plt1.imshow(binary_and_x_grad, cmap="gray")
    # plt1.set_title("X, Grad, And")
    # plt2.imshow(binary_or_x_grad, cmap="gray")
    # plt2.set_title("X, Grad, Or")
    # plt3.imshow(binary_and_mag_grad, cmap="gray")
    # plt3.set_title("Mag, Grad, And")
    # plt4.imshow(binary_or_mag_grad, cmap="gray")
    # plt4.set_title("Mag, Grad, Or")
    # plt.show()
    
    return binary_and_x_grad, binary_and_mag_grad

def threshold_combined(thresholded_hsl, thresholded_sobel):
    # And
    combined_and = np.zeros_like(thresholded_hsl)
    combined_and[(thresholded_hsl == 1) & (threshold_sobel == 1)] = 1

    # Or
    combined_or = np.zeros_like(thresholded_hsl)
    combined_or[(thresholded_hsl == 1) | (thresholded_sobel == 1)] = 1

    return combined_and, combined_or

if __name__ == "__main__":
    test_img_path = "./test_images/test6.jpg"
    test_img = cv2.imread(test_img_path)

    # Threshold the image (HSL)
    thresholded_hsl_60 = threshold_hsl(test_img, 60)
    thresholded_hsl_80 = threshold_hsl(test_img, 80)
    thresholded_hsl_100 = threshold_hsl(test_img, 100)

    # Threshold the image (Sobel and gradient)
    thresholded_sobel_and_x, thresholded_sobel_and_mag = threshold_sobel(test_img, 
                                                                         sobel_kernel=15, 
                                                                         mag_thresh=(50, 190), 
                                                                         grad_thresh=(0.7, 1.2))

    # Combine HSL and Sobel threshold
    hsl_and_x, hsl_or_x = threshold_combined(thresholded_hsl_100, thresholded_sobel_and_x) 
    hsl_and_mag, hsl_or_mag = threshold_combined(thresholded_hsl_100, thresholded_sobel_and_mag) 

    # Stack the binaries
    color_binary = np.dstack((np.zeros_like(thresholded_hsl_100), 
                             thresholded_hsl_100, 
                             thresholded_sobel_and_mag)) * 255


    # Show the images
    fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2, figsize=(20,10))
    plt1.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt1.set_title("Original")
    plt3.imshow(hsl_or_x, cmap="gray")
    plt3.set_title("HSL or X")
    # plt3.imshow(hsl_and_mag, cmap="gray")
    # plt3.set_title("HSL and Mag")
    plt4.imshow(hsl_or_mag, cmap="gray")
    plt4.set_title("HSL or Mag")
    plt.show()