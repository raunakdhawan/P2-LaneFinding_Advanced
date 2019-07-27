import cv2
import numpy as np
from matplotlib import pyplot as plt


def threshold_hsl(raw_image, thresh_min=100):
    '''
    Options for thresholding
    1. Using HSL - Insted of using gray scale color space, will be using the HSL
    2. Using the Sobel with gradient
    3. Using the canny edge detection
    4. Using histogram
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

if __name__ == "__main__":
    test_img_path = "./test_images/test5.jpg"
    test_img = cv2.imread(test_img_path)

    # Threshold the image
    thresholded = threshold_hsl(test_img, 60)
    thresholded_1 = threshold_hsl(test_img, 80)
    thresholded_2 = threshold_hsl(test_img, 100)

    # Show the images
    fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2, figsize=(20,10))
    plt1.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt2.imshow(thresholded, cmap="gray")
    plt3.imshow(thresholded_1, cmap="gray")
    plt4.imshow(thresholded_2, cmap="gray")
    plt.show()