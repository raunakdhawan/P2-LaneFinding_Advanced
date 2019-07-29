import numpy as np
import cv2

def draw_lane_lines(img, warped, left_points, right_points, M):
    # Create canvas
    canvas = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    canvas_color = np.dstack((canvas, canvas, canvas))  # Make it 3D for RGB

    # Get the points in a format that is usable in cv2.fillPoly -> np.array([[y, x]])
    pts_left = np.array([np.transpose(np.vstack([left_points[0], left_points[1]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_points[0], right_points[1]])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the blank image
    cv2.fillPoly(canvas_color, np.array( [pts], dtype=np.int32 ), (255, 0, 0))

    # Unwarp the image
    unwarped = cv2.warpPerspective(canvas_color, M, (img.shape[1], img.shape[0])) 

    # Combine the result with the original image
    combined = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

    # Return a BGR image, needs to be converted to RGB when using matplotlib
    return combined