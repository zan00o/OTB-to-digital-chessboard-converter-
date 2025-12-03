#!/bin/bash
# WSL/Linux equivalent of build_all.cmd

POSITIONS="input_imgs"
CORNERS="data/corners"
DATASET="data/dataset"

# Loop through each subfolder
for folder in "${POSITIONS}"/*/; do
    folder_name=$(basename "$folder")
    fen_file="${folder}/fen_list.csv"
    
    if [ -f "$fen_file" ]; then
        echo "Building dataset for folder ${folder_name}..."
        python -m src.build_dataset \
            --folder "$folder" \
            --corners-dir "${CORNERS}/${folder_name}" \
            --fen-file "$fen_file" \
            --dataset-root "$DATASET" \
            --img-size 96
    else
        echo "Skipping ${folder_name} (no fen_list.csv found)"
    fi
done

echo "All folders processed."