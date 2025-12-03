"""
squares.py
Functions for splitting a top-down image of a chessboard into its 64 squares.
Necessary once OpenCV has used homography to produce a top-down view of the board, so we can
then extract individual square images for classification.

Board is assumed to be correctly oriented with a8 at top-left after warp
(ensured by annotating corners in chess order: a8, h8, h1, a1).
"""

import numpy as np

# Split the top-down board image into its 64 square crops
def split_squares(topdown_view, pad=10, height_ratio=1.0):
    # topdown_view: square image (800x800 from warp)
    # assumes input is square (N x N) 
    N = topdown_view.shape[0]
    # size of each square
    cell = N // 8
    crops = []
    
    # Calculate how much extra height to capture above the square
    extra_height = int(cell * (height_ratio - 1.0))
    
    # iterate over 8 rows, 8 columns
    # Row 0 = rank 8, Row 7 = rank 1
    # Col 0 = file a, Col 7 = file h
    # Index 0 = a8, Index 7 = h8, Index 8 = a7, ..., Index 63 = h1
    for r in range(8):
        for c in range(8):
            # coord x0, y0, x1, y1 with padding
            x0 = max(c*cell + pad, 0)
            x1 = min((c+1)*cell - pad, N)
            
            # For height, extend upward to capture more of the piece
            # y0 starts higher (smaller value) to capture area above the square
            y0 = max(r*cell + pad - extra_height, 0)
            y1 = min((r+1)*cell - pad, N)
            
            crops.append(topdown_view[y0:y1, x0:x1])
    return crops
