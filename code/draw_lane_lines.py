import numpy as np
import cv2

def color_lane(img, warped_binary, left_points, right_points, left_curve, right_curve, offset, M):
    # Create canvas
    canvas = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    canvas_color = np.dstack((canvas, canvas, canvas))  # Make it 3D for RGB

    # Get the points in a format that is usable in cv2.fillPoly -> np.array([[y, x]])
    pts_left = np.array([np.transpose(np.vstack([left_points[0], left_points[1]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_points[0], right_points[1]])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw and color the lane polygon onto the blank image
    cv2.fillPoly(canvas_color, np.array([pts], dtype=np.int32), (255, 0, 0))

    # Plot the fitted polynomial
    left_poly_pts = np.int32(np.column_stack((left_points[0], left_points[1])))
    right_poly_pts = np.int32(np.column_stack((right_points[0], right_points[1])))
    cv2.polylines(canvas_color, [left_poly_pts], False, (0, 0, 255), 30)
    cv2.polylines(canvas_color, [right_poly_pts], False, (0, 255, 0), 30)

    # Unwarp the image
    unwarped = cv2.warpPerspective(canvas_color, M, (img.shape[1], img.shape[0])) 

    # Combine the result with the original image
    combined = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

    # Add the text
    # Write text on the image
    with_lanes_text = combined.copy()
    cv2.putText(combined, 
                "Curvature (Left Lane): {:.2f} m".format(left_curve), 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255,255,255), 
                2)
    cv2.putText(combined, 
                "Curvature (Right Lane): {:.2f} m".format(right_curve),
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1., 
                (255,255,255), 
                2)
    cv2.putText(combined, 
                    "Car Offset : {:.2f} m".format(offset),
                    (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1., 
                    (255,255,255), 
                    2)

    # # Add the warped image on top
    # new_warped = cv2.resize(warped_binary, (280, 180))
    # new_warped[new_warped == 1] = 255
    # new_warped = np.dstack((new_warped, new_warped, new_warped))
    # combined[20:20+new_warped.shape[0], 950:950+new_warped.shape[1]] = new_warped

    # Return a BGR image, needs to be converted to RGB when using matplotlib
    return combined