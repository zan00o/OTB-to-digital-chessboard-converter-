"""
make_fen_list.py
Generate fen_list.csv files for each input image folder.
fen_list.csv format:
image_filename,fen_string
Where FEN string is the first part (just piece positions) of a full Fensworth-Edwards Notation string.
Allows us to associate each image with its correct board position during dataset building.

"""

import pathlib, os 
dir = r"C:\Users\happy\Desktop\notes\Classes\ENGR413\Chess2FEN\input_imgs"
for i in range (1, 11):
    fen = (pathlib.Path(dir)/f"{i}/fen.txt").read_text()
    with open(pathlib.Path(dir)/f"{i}/fen_list.csv", "w", newline="") as f:
        for p in os.listdir(pathlib.Path(dir)/f"{i}"):
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                f.write(f"{p},{fen}\n")
    print(f"wrote {pathlib.Path(dir)/f'{i}/fen_list.csv'}")