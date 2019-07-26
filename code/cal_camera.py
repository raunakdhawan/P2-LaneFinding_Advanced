import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def cal_camera(path_cal_imgs, nx, ny):
    # Calibration Images List
    cal_images = glob.glob(path_cal_imgs)

    # Create Object Points
    obj_p = np.zeros((nx*ny, 3), np.float32)
    obj_p[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    obj_points = []
    img_points = []

    img_size = (0, 0)

    # Loop through the list and search for chess board corners
    for inx, image_path in enumerate(cal_images):
        # Load the image
        image = cv2.imread(image_path)
        img_size = (image.shape[1], image.shape[0])

        # Convert to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

        if ret == True:
            obj_points.append(obj_p)
            img_points.append(corners)

    # Calculate the calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    return ret, mtx, dist, rvecs, tvecs

if __name__ == "__main__":
    # Calibration of Camera
    path_imgs = "./camera_cal/calibration*.jpg"
    num_x = 9
    num_y = 5

    # Calculate calibration parameters
    ret, mtx, dist, rvecs, tvecs = cal_camera(path_imgs, num_x, num_y)

    # Test on an image
    img = cv2.imread("./camera_cal/calibration1.jpg")
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

    # Pickle the results
    pickle_file_path = "./code/calibration.p"
    calibration = {}
    calibration["mtx"] = mtx
    calibration["dist"] = dist
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(calibration, pickle_file)
