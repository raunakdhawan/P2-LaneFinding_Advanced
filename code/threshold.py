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

    return hsl_threshold

if __name__ == "__main__":
    test_img_path = "./test_images/test3.jpg"
    test_img = cv2.imread(test_img_path)

    # Print Image size
    print(test_img.shape)

    # Threshold the image
    thresholded = threshold_hsl(test_img, 60)

    # Show the images
    fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(20,10))
    plt1.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt2.imshow(thresholded, cmap="gray")
    plt.show()