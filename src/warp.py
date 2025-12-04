"""
warp.py
Functions for warping a chessboard image to a top-down view.
Used in annotate_corners.py, infer_image.py, and build_dataset.py.
Using the points from the corner annotation step, we compute a homography to warp
the input image to a square top-down view of the chessboard.

Corners are expected in CHESS ORDER: a8, h8, h1, a1
This maps to destination: TL, TR, BR, BL
"""

import cv2
import numpy as np

# Warp the board to a top-down view using the given corners
# corners_xy must be in chess order: [a8, h8, h1, a1]
def warp_board(img_bgr, corners_xy, out_size=800):
    # takes corners as list or np array of (x,y) points
    # Expected order: a8 (TL), h8 (TR), h1 (BR), a1 (BL)
    corners_xy = np.array(corners_xy, dtype=np.float32)
    if corners_xy.shape != (4,2):
        raise ValueError("corners_xy must be (4,2).")
    
    # Corners are already in chess order: a8, h8, h1, a1
    # This maps directly to: TL, TR, BR, BL of the output
    # define destination points for homography
    # a8 -> (0,0), h8 -> (out_size-1,0), h1 -> (out_size-1,out_size-1), a1 -> (0,out_size-1)
    dst = np.float32([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]])
    
    # use CV2 to compute homography and warp
    H = cv2.getPerspectiveTransform(corners_xy, dst)
    topdown = cv2.warpPerspective(img_bgr, H, (out_size, out_size))
    return topdown, H

# Keep order_corners for backwards compatibility if needed elsewhere
def order_corners(pts):
    """Legacy function - orders corners by image position (TL, TR, BR, BL)"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)
