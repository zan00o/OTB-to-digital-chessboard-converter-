"""
warp.py
Functions for ordering corners and warping a chessboard image to a top-down view.
Used in annotate_corners.py, infer_image.py, and build_dataset.py.
Using the points from the corner annotation step, we compute a homography to warp
the input image to a square top-down view of the chessboard.
"""

import cv2
import numpy as np

# Order corners as Top Left, Top Right, Bottom Right, Bottom Left
# Based on perspective of image, not chessboard orientation 
# (i.e , TL is where the top-left corner appears in the image)
def order_corners(pts):
    # corner coordinates from annotation
    # cast to np array, 2-D float32
    pts = np.array(pts, dtype=np.float32)
    # Calculate sums and differences
    # Sum of coordinates gives TL (min) and BR (max)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    # ex:
    # pts = [(400, 100), (100, 100), (400, 400), (100, 400)]
    # order_corners(pts)
    # Point	    (x, y)	x+y	    x−y
    # (400,100)	500	    300	
    # (100,100)	200	    0	
    # (400,400)	800	    0	
    # (100,400)	500	    -300
    # based off assumption that origin (0, 0) is top-left, common for CV2
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    # TL would have smallest sum, BR largest sum
    # TR would have smallest difference, BL largest difference
    return np.array([tl, tr, br, bl], dtype=np.float32)

# Warp the board to a top-down view using the given corners
def warp_board(img_bgr, corners_xy, out_size=800):
    # takes corners as list or np array of (x,y) points
    corners_xy = np.array(corners_xy, dtype=np.float32)
    if corners_xy.shape != (4,2):
        raise ValueError("corners_xy must be (4,2).")
    # get TL, TR, BR, BL order
    corners_xy = order_corners(corners_xy)
    # define destination points for homography
    # TL = (0,0), TR = (out_size-1,0), BR = (out_size-1,out_size-1), BL = (0,out_size-1)
    dst = np.float32([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]])
    # use CV2 to compute homography and warp
    H = cv2.getPerspectiveTransform(corners_xy, dst)
    topdown = cv2.warpPerspective(img_bgr, H, (out_size, out_size))
    return topdown, H
