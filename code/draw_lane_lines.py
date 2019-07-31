import numpy as np
import cv2

def color_lane(img, warped_binary, left_points, right_points, left_lane_pixels, right_lane_pixels, M):
    # Create canvas
    canvas = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    canvas_color = np.dstack((canvas, canvas, canvas))  # Make it 3D for RGB

    # Get the points in a format that is usable in cv2.fillPoly -> np.array([[y, x]])
    pts_left = np.array([np.transpose(np.vstack([left_points[0], left_points[1]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_points[0], right_points[1]])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw and color the lane onto the blank image
    cv2.fillPoly(canvas_color, np.array([pts], dtype=np.int32), (255, 0, 0))
    canvas_color[left_lane_pixels[1], left_lane_pixels[0]] = (0, 0, 255)
    canvas_color[right_lane_pixels[1], right_lane_pixels[0]] = (0, 255, 0)

    # Unwarp the image
    unwarped = cv2.warpPerspective(canvas_color, M, (img.shape[1], img.shape[0])) 

    # Combine the result with the original image
    combined = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

    # Return a BGR image, needs to be converted to RGB when using matplotlib
    return combined