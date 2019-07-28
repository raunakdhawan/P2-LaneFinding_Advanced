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
    
    # Calculate sobel magnitude and scale
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))

    # Binary output
    binary_output_mag = np.zeros_like(scaled_sobel)
    binary_output_mag[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # Take the absolute value of the x and y gradients
    abs_x = np.abs(sobelx)
    abs_y = np.abs(sobely)
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    atan = np.arctan2(abs_y, abs_x)
    
    # Create a binary mask where direction thresholds are met
    binary_output_grad = np.zeros_like(atan)
    binary_output_grad[(atan >= grad_thresh[0]) & (atan <= grad_thresh[1])] = 1

    # Combine the binary outputs
    binary_and = np.zeros_like(binary_output_grad)
    binary_and[(binary_output_mag == 1) & (binary_output_grad == 1)] = 1
    
    binary_or = np.zeros_like(binary_output_grad)
    binary_or[(binary_output_mag == 1) | (binary_output_grad == 1)] = 1

    # fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2)
    # plt1.imshow(scaled_sobel, cmap="gray")
    # plt2.imshow(binary_output_mag, cmap="gray")
    # plt3.imshow(atan, cmap="gray")
    # plt4.imshow(binary_output_grad, cmap="gray")
    # plt.show()
    
    return binary_and, binary_or

if __name__ == "__main__":
    test_img_path = "./test_images/test5.jpg"
    test_img = cv2.imread(test_img_path)

    # Threshold the image (HSL)
    thresholded_hsl = threshold_hsl(test_img, 60)
    thresholded_1 = threshold_hsl(test_img, 80)
    thresholded_2 = threshold_hsl(test_img, 100)

    # Threshold the image (Sobel and gradient)
    thresholded_sobel_and, thresholded_sobel_or = threshold_sobel(test_img, 
                                                                    sobel_kernel=15, 
                                                                    mag_thresh=(50, 190), 
                                                                    grad_thresh=(0.7, 1.2))

    # Show the images
    fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2, figsize=(20,10))
    plt1.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt2.imshow(thresholded_hsl, cmap="gray")
    plt3.imshow(thresholded_sobel_and, cmap="gray")
    plt4.imshow(thresholded_sobel_or, cmap="gray")
    plt.show()