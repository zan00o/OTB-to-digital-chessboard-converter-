"""
Interactive piece cropper - draw a box around each piece
Keys:
  - Click and drag to draw crop box
  - ENTER/SPACE: Save crop and move to next image
  - R: Reset crop box
  - S: Skip this image
  - Q: Quit
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

class PieceCropper:
    def __init__(self, image_path, piece_class, output_dir, target_size=96):
        self.image_path = Path(image_path)
        self.piece_class = piece_class
        self.output_dir = Path(output_dir) / piece_class
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        
        self.img = cv2.imread(str(image_path))
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize if too large for display
        self.display_scale = 1.0
        max_display = 1200
        h, w = self.img.shape[:2]
        if max(h, w) > max_display:
            self.display_scale = max_display / max(h, w)
        
        self.display_img = cv2.resize(self.img, None, fx=self.display_scale, fy=self.display_scale)
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.temp_img = self.display_img.copy()
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_pt = (x, y)
                self.temp_img = self.display_img.copy()
                cv2.rectangle(self.temp_img, self.start_pt, self.end_pt, (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_pt = (x, y)
            self.temp_img = self.display_img.copy()
            cv2.rectangle(self.temp_img, self.start_pt, self.end_pt, (0, 255, 0), 2)
    
    def run(self):
        window_name = f"Crop {self.piece_class} - {self.image_path.name}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            cv2.imshow(window_name, self.temp_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                self.start_pt = None
                self.end_pt = None
                self.temp_img = self.display_img.copy()
                print("Reset crop box")
            
            elif key == ord('s'):  # Skip
                print(f"Skipped {self.image_path.name}")
                cv2.destroyWindow(window_name)
                return False
            
            elif key == ord('q'):  # Quit
                print("Quitting...")
                cv2.destroyWindow(window_name)
                return None
            
            elif key in [13, 32]:  # ENTER or SPACE - save
                if self.start_pt and self.end_pt:
                    success = self.save_crop()
                    cv2.destroyWindow(window_name)
                    return success
                else:
                    print("Draw a crop box first!")
        
        cv2.destroyWindow(window_name)
        return False
    
    def save_crop(self):
        # Convert display coordinates back to original image coordinates
        x1 = int(self.start_pt[0] / self.display_scale)
        y1 = int(self.start_pt[1] / self.display_scale)
        x2 = int(self.end_pt[0] / self.display_scale)
        y2 = int(self.end_pt[1] / self.display_scale)
        
        # Ensure correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Crop and resize
        crop = self.img[y1:y2, x1:x2]
        if crop.size == 0:
            print("Invalid crop!")
            return False
        
        resized = cv2.resize(crop, (self.target_size, self.target_size))
        
        # Save with unique name
        existing = list(self.output_dir.glob(f"{self.image_path.stem}_*.png"))
        idx = len(existing)
        output_path = self.output_dir / f"{self.image_path.stem}_{idx:03d}.png"
        
        cv2.imwrite(str(output_path), resized)
        print(f"✓ Saved: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Interactive piece cropper")
    parser.add_argument("--folder", required=True, help="Folder with images")
    parser.add_argument("--piece", required=True, help="Piece class (e.g., white_pawn)")
    parser.add_argument("--output", default="data/dataset/raw", help="Output dataset root")
    parser.add_argument("--size", type=int, default=96, help="Target size (default: 96)")
    args = parser.parse_args()
    
    folder = Path(args.folder)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {folder}")
        return
    
    print(f"Found {len(image_files)} images")
    print("\nControls:")
    print("  - Click and drag to draw crop box")
    print("  - ENTER/SPACE: Save crop and next image")
    print("  - R: Reset crop box")
    print("  - S: Skip this image")
    print("  - Q: Quit\n")
    
    saved_count = 0
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")
        
        cropper = PieceCropper(img_path, args.piece, args.output, args.size)
        result = cropper.run()
        
        if result is None:  # Quit
            break
        elif result:  # Saved
            saved_count += 1
    
    print(f"\n✓ Done! Saved {saved_count} crops")


if __name__ == "__main__":
    main()