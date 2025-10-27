# Chess2FEN (Per-Square Baseline)

A minimal **per-square** pipeline that converts photos of your **own chessboard** into a FEN **piece placement** string (field #1).  
It’s designed for a class checkpoint or demo — simple, fast, and easy to extend.

---

## 🧠 Overview

Workflow:

1. **Annotate board corners** for each image (once per photo or batch).
2. **Warp** each board → split into **64 square crops**.
3. **Label** each crop automatically from a known FEN string.
4. **Train** a small CNN classifier (13 classes: `empty` + 12 pieces).
5. **Infer** a FEN from new board photos.

---

## ⚙️ Setup

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\activate    # (Windows)
pip install -r requirements.txt
```

## 🪟 Step 1 — Annotate corners (batch or single)

### Annotate _all_ images in a folder

`python -m src.annotate_corners --folder .\input_imgs --out .\data\corners`

- Opens each image one at a time.
- Click **Top-Left → Top-Right → Bottom-Right → Bottom-Left**.
- Keys:
    - **s** = save corners for this image → auto-advance
    - **r** = reset
    - **f** = flip preview
    - **q** = quit early

Saved files:
`data/corners/inputImg01.json data/corners/inputImg02.json ...`

### Annotate just one image

`python -m src.annotate_corners --image .\input_imgs\inputImg01.jpg --out .\data\corners\inputImg01.json`

---
## ♟️ Step 2 — Build a small dataset

Use known positions (for example, the starting position) and type their FENs.  
This automatically slices 64 crops and saves them to class folders.

```
python -m src.build_dataset `   
--image .\input_imgs\inputImg01.jpg `   
--corners .\data\corners\inputImg01.json `  
--fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" `  
--dataset-root .\data\dataset
```

Repeat for ~10–30 varied positions to get 640–1920 labeled crops.
Resulting folder structure:

```
data/  
	└─ dataset/      
		└─ raw/          
			├─ empty/         
			├─ white_pawn/         
			├─ black_pawn/          
			├─ white_king/         
```
---

## 🧠 Step 3 — Train the classifier (13 classes)

```
python -m src.train_classifier `  
	--dataset-root .\data\dataset\raw `
	--out .\models\classifier.keras
```

- Uses a small CNN built with Keras/TensorFlow.
- Automatically saves `.classes.json` (class order).

---

## 🔍 Step 4 — Inference: Image → FEN

Predict the piece placement on a new board photo:

```
python -m src.infer_image `   
--image .\input_imgs\new_board.jpg `   
--corners .\data\corners\inputImg01.json `  
--model .\models\classifier.keras
```

Example output:

`FEN (placement): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR FEN (full)     : rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1`

If the board appears upside down, add:

`--flip180`

---

## 🧩 Tips

- Run **all commands from the project root** (where `src/` is located).
- Ensure `src\__init__.py` exists (even if empty).
- Use the venv’s Python (`.\.venv\Scripts\python.exe`) instead of Anaconda’s.
- Each image’s corner JSON should match the camera viewpoint used for that photo.
- For class checkpoints, it’s fine to process one board and one piece set.
    

---

## 🧭 Next Steps (after checkpoint)

- Replace manual corner clicks with an **automatic corner detector** (YOLO or pose-net).
- Collect data from multiple boards/pieces for generalization.
- Extend inference to **video** using temporal smoothing and hand detection.