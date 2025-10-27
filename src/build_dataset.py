import argparse, json, os, cv2, numpy as np
from pathlib import Path
from .warp import warp_board
from .squares import split_squares, maybe_flip_180
from .fen_utils import LABELS

FEN_TO_LABEL = {
    'K':'white_king','Q':'white_queen','R':'white_rook','B':'white_bishop','N':'white_knight','P':'white_pawn',
    'k':'black_king','q':'black_queen','r':'black_rook','b':'black_bishop','n':'black_knight','p':'black_pawn'
}

def parse_fen_placement(placement):
    rows = placement.split('/')
    if len(rows) != 8:
        raise ValueError("FEN placement must have 8 ranks")
    labels = []
    for r in rows:
        for ch in r:
            if ch.isdigit():
                labels.extend(["empty"] * int(ch))
            else:
                labels.append(FEN_TO_LABEL[ch])
    if len(labels) != 64:
        raise ValueError("Decoded labels != 64")
    return labels

def process_one(image_path, corners_path, fen_str, out_root, img_size=96):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Could not read image {image_path}")
        return
    corners = np.array(json.load(open(corners_path)), dtype=np.float32)
    topdown, _ = warp_board(img, corners, out_size=800)
    topdown = maybe_flip_180(topdown)  # ensure A1 is dark in bottom-left to match FEN order
    crops = split_squares(topdown, pad=2)
    labels = parse_fen_placement(fen_str.strip())

    for lab in LABELS:
        (out_root / lab).mkdir(parents=True, exist_ok=True)

    for i, (crop, lab) in enumerate(zip(crops, labels)):
        out_path = out_root / lab / f"{Path(image_path).stem}_{i:02d}.png"
        resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), resized)

    print(f"✅ {image_path.name} → {len(crops)} squares saved")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, help="Single image path")
    ap.add_argument("--corners", type=str, help="Path to its corner JSON")
    ap.add_argument("--fen", type=str, help="FEN piece placement string")
    ap.add_argument("--folder", type=str, help="Folder with input images")
    ap.add_argument("--corners-dir", type=str, help="Folder with corner JSONs")
    ap.add_argument("--fen-file", type=str, help="Text file: filename,FEN per line")
    ap.add_argument("--dataset-root", required=True, type=str)
    ap.add_argument("--img-size", type=int, default=96)
    args = ap.parse_args()

    out_root = Path(args.dataset_root) / "raw"

    # Option A — single image mode
    if args.image:
        if not (args.corners and args.fen):
            raise SystemExit("For single-image mode, supply --image, --corners, and --fen")
        process_one(Path(args.image), Path(args.corners), args.fen, out_root, img_size=args.img_size)
        return

    # Option B — folder mode
    if not (args.folder and args.corners_dir and args.fen_file):
        raise SystemExit("For folder mode, supply --folder, --corners-dir, and --fen-file")

    folder = Path(args.folder)
    corners_dir = Path(args.corners_dir)
    fen_lines = [l.strip() for l in open(args.fen_file) if l.strip()]
    fen_map = {}
    for line in fen_lines:
        try:
            name, fen_str = line.split(",", 1)
            fen_map[name.strip()] = fen_str.strip()
        except ValueError:
            print(f"⚠️ Skipping malformed line: {line}")

    for img_path in sorted(folder.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        name = img_path.name
        fen_str = fen_map.get(name)
        if not fen_str:
            print(f"⚠️ No FEN found for {name}, skipping.")
            continue
        corners_path = corners_dir / f"{img_path.stem}.json"
        if not corners_path.exists():
            print(f"⚠️ Missing corners for {name}, skipping.")
            continue
        process_one(img_path, corners_path, fen_str, out_root, img_size=args.img_size)

    print("✅ Dataset build complete.")

if __name__ == "__main__":
    main()
