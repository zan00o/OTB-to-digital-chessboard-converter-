"""
annotate_corners.py
Tool for manually annotating the four corners of a chessboard in an image.
User clicks on the presented image in the order (From perspective of image):
Top-Left, Top-Right, Bottom-Right, Bottom-Left
Saves the corner coordinates to a JSON file for later use in warping.
"""

import argparse, json, cv2, numpy as np
from pathlib import Path
from .warp import order_corners

HELP = "Click 4 corners: TL, TR, BR, BL. [s]=save, [r]=reset, [f]=flip, [q]=quit"

def on_mouse(event, x, y, flags, param):
    pts = param["pts"]
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

def annotate_image(img_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        return False

    # Resize for display
    scale = 0.7
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    pts = []
    flip_preview = False
    cv2.namedWindow("corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("corners", on_mouse, {"pts": pts})

    while True:
        vis = img.copy()
        if flip_preview:
            vis = vis[::-1, ::-1]

        cv2.putText(vis, HELP, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 6, (0,0,255), -1)
            cv2.putText(vis, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("corners", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            return "quit"
        elif k == ord('r'):
            pts.clear()
        elif k == ord('f'):
            flip_preview = not flip_preview
        elif k == ord('s'):
            if len(pts) != 4:
                print("Need exactly 4 points.")
                continue
            ordered = order_corners(pts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(ordered.tolist(), f, indent=2)
            print(f"✅ Saved corners to {out_path}")
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
        print("All done ✅")
    elif args.image:
        annotate_image(Path(args.image), Path(args.out))
        cv2.destroyAllWindows()
    else:
        raise SystemExit("Provide either --folder or --image")

if __name__ == "__main__":
    main()
