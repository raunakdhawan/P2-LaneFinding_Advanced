import numpy as np
from draw_lane_lines import color_lane
from histogram_lane_pixels import search_around_poly


def sanity_check(raw_img, lane, M):
    # Check the distance between the two polynomials
    lane_width = np.average(lane.right_poly_pts[0]) - np.average(lane.left_poly_pts[0])

    if lane_width > 850 or lane_width < 700:
        print("Lane not detected, with width {:.2f}".format(lane_width))
        
        # Update the lane class
        lane.left_poly_pts = lane.left_poly_pts_previous
        lane.right_poly_pts = lane.right_poly_pts_previous
        lane.detected = False

    else:
        # Update the lane class
        lane.left_poly_pts_previous = lane.left_poly_pts
        lane.right_poly_pts_previous = lane.right_poly_pts
        lane.detected = True
    
    # Draw the image
    lane.detected_lane_img = color_lane(raw_img, 
                                        lane.warped_binary, 
                                        lane.left_poly_pts, 
                                        lane.right_poly_pts, 
                                        lane.left_curve_rad, 
                                        lane.right_curve_rad,
                                        lane.offset,
                                        M)

    return lane