"""
squares.py
Functions for splitting a top-down image of a chessboard into its 64 squares,
and for checking/flipping orientation based on the color of the a1 square.
Necessary once OpenCV has used homography to produce a top-down view of the board, so we can
then extract individual square images for classification.
"""

import numpy as np

# Split the top-down board image into its 64 square crops
def split_squares(topdown_view, pad=2):
    # topdown_view: square image (800x800 from warp)
    # assumes input is square (N x N) 
    N = topdown_view.shape[0]
    # size of each square
    cell = N // 8
    crops = []
    # iterate over 8 rows, 8 columns
    # 64 crops (for squares a1 to h8)
    for r in range(8):
        for c in range(8):
            # coord x0, y0, x1, y1 with padding
            # basically split at 100 * iteration + pad
            x0 = max(c*cell + pad, 0); y0 = max(r*cell + pad, 0)
            x1 = min((c+1)*cell - pad, N); y1 = min((r+1)*cell - pad, N)
            crops.append(topdown_view[y0:y1, x0:x1])
    return crops

# A1 hueristic: for a correctly oriented board, a1 is a dark square
def is_a1_dark(topdown_view):
    N = topdown_view.shape[0]
    cell = N // 8
    a1 = topdown_view[7*cell:(8*cell), 0:cell]
    # compare mean brightness of a1 to overall mean
    # if a1 is darker than overall mean, assume correct orientation
    return a1.mean() < topdown_view.mean()

# Flip the board 180deg if a1 is not dark (or if forced)
def maybe_flip_180(topdown_view, force_flip=False):
    img = topdown_view.copy()
    if force_flip:
        # force flip regardless of a1 color
        return np.ascontiguousarray(img[::-1, ::-1])
    if not is_a1_dark(img):
        # flip if a1 is not dark
        # img [::-1, ::-1] reverses both axes by reverse-slicing
        # as img is a numpy array of (Height, Width, [color] channels)
        return np.ascontiguousarray(img[::-1, ::-1])
    return img
