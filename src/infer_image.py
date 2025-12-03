import argparse, json, numpy as np, cv2, tensorflow as tf
from pathlib import Path
from .warp import warp_board
from .squares import split_squares
from .fen_utils import LABELS, grid_to_fen_placement, full_fen_from_placement

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--corners", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--img-width", type=int, default=96)
    ap.add_argument("--img-height", type=int, default=192, help="Height of crops (default 192 for 2:1 ratio)")
    ap.add_argument("--height-ratio", type=float, default=2.0, help="How much taller to crop (2.0 = extend one cell up)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    corners = np.array(json.load(open(args.corners)), dtype=np.float32)

    model = tf.keras.models.load_model(args.model)
    classes_path = Path(args.model).with_suffix(".classes.json")
    if classes_path.exists():
        class_names = json.load(open(classes_path))
    else:
        class_names = LABELS

    topdown, _ = warp_board(img, corners, out_size=800)
    # Use taller crops to match training data
    crops = split_squares(topdown, pad=10, height_ratio=args.height_ratio)

    # Resize to (width, height) - cv2.resize takes (width, height)
    batch = np.stack([cv2.resize(c, (args.img_width, args.img_height)) for c in crops], axis=0)
    batch = batch.astype(np.float32)/255.0
    probs = model.predict(batch, verbose=0)
    ids = probs.argmax(axis=1).tolist()
    pred_labels = [class_names[i] for i in ids]

    placement = grid_to_fen_placement(pred_labels)
    full_fen = full_fen_from_placement(placement, side_to_move="w", castling="-", ep="-", halfmove="0", fullmove="1")
    print("FEN (placement):", placement)
    print("FEN (full)     :", full_fen)

if __name__ == "__main__":
    main()
