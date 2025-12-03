"""
annotate_corners.py
Tool for manually annotating the four corners of a chessboard in an image.
User clicks on the presented image in CHESS ORDER:
a8 (top-left of board), h8, h1, a1 (bottom-left of board)
Saves the corner coordinates to a JSON file for later use in warping.
"""

import argparse, json, cv2, numpy as np
from pathlib import Path

HELP = "Click 4 corners: a8, h8, h1, a1. [s]=save, [r]=reset, [q]=quit"

def on_mouse(event, x, y, flags, param):
    pts = param["pts"]
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

def annotate_image(img_path: Path, out_path: Path):
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"Could not read {img_path}")
        return False

    # Resize for display
    scale = 0.7
    h, w = img_orig.shape[:2]
    img = cv2.resize(img_orig, (int(w * scale), int(h * scale)))

    pts = []
    cv2.namedWindow("corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("corners", on_mouse, {"pts": pts})

    while True:
        vis = img.copy()

        cv2.putText(vis, HELP, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3, cv2.LINE_AA)
        # Label points with chess square names
        labels = ["a8", "h8", "h1", "a1"]
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 6, (0,0,255), -1)
            label = labels[i] if i < 4 else str(i+1)
            cv2.putText(vis, label, (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("corners", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            return "quit"
        elif k == ord('r'):
            pts.clear()
        elif k == ord('s'):
            if len(pts) != 4:
                print("Need exactly 4 points.")
                continue
            # Scale points back to original image coordinates
            # Points are in order: a8, h8, h1, a1 (user clicked in this order)
            scaled_pts = [[p[0] / scale, p[1] / scale] for p in pts]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(scaled_pts, f, indent=2)
            print(f"Saved corners to {out_path}")
            return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, help="Folder containing input images")
    ap.add_argument("--image", type=str, help="Single image path (optional)")
    ap.add_argument("--out", required=True, type=str, help="Output folder or file")
    args = ap.parse_args()

    if args.folder:
        img_paths = sorted(
            [p for p in Path(args.folder).iterdir()
             if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        )
        if not img_paths:
            raise SystemExit("No images found in folder.")
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            out_path = out_dir / f"{img_path.stem}.json"
            print(f"\n=== {img_path.name} ===")
            result = annotate_image(img_path, out_path)
            if result == "quit":
                break

        cv2.destroyAllWindows()
        print("All done")
    elif args.image:
        annotate_image(Path(args.image), Path(args.out))
        cv2.destroyAllWindows()
    else:
        raise SystemExit("Provide either --folder or --image")

if __name__ == "__main__":
    main()
