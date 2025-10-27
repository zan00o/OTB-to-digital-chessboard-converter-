"""
fen_utils.py
Utilities for handling FEN strings and piece labels.
Objects for FEN notation& labels
Functions for converting between grid labels and FEN placement strings.
"""


LABELS = [
    "empty",
    "white_king","white_queen","white_rook","white_bishop","white_knight","white_pawn",
    "black_king","black_queen","black_rook","black_bishop","black_knight","black_pawn"
]

PIECE_CODE = {
    'white_king':'K','white_queen':'Q','white_rook':'R','white_bishop':'B','white_knight':'N','white_pawn':'P',
    'black_king':'k','black_queen':'q','black_rook':'r','black_bishop':'b','black_knight':'n','black_pawn':'p'
}

def grid_to_fen_placement(grid_labels):
    # grid labels from output of infer_image.py
    if len(grid_labels) != 64:
        raise ValueError("Need 64 labels (8x8).")
    rows = []
    # iterate over all 8 ranks
    for r in range(8):  
        # r=0 => rank 8, max rank system
        # FEN notation goes rank8/.../rank1
        row = []
        empty_run = 0
        # all 8 files in this rank
        for c in range(8):
            # r*8 + c gives index in 0-63
            lab = grid_labels[r*8 + c]
            if lab == "empty" or lab is None:
                # FEN counts empty squares so like if rook on a1, otehr rook on h1, then
                # notation would be R6R for that rank
                # this is the counter for that
                empty_run += 1
            else:
                if empty_run > 0:
                    # append number of empty squares before piece code
                    row.append(str(empty_run))
                    # reset counter
                    empty_run = 0
                row.append(PIECE_CODE[lab])
        if empty_run > 0:
            row.append(str(empty_run))
        rows.append("".join(row) if row else "8")
    return "/".join(rows)

# full FEN string
# havent actually implemented stuff like castling, en passant, etc
# and im not sure if thats possible from still-frames alone
def full_fen_from_placement(placement, side_to_move="w", castling="-", ep="-", halfmove="0", fullmove="1"):
    return f"{placement} {side_to_move} {castling} {ep} {halfmove} {fullmove}"
