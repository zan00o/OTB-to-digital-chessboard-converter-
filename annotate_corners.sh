#!/bin/bash
# WSL/Linux equivalent of annotate_corners.cmd

POSITIONS="input_imgs/new"
CORNERS="data/corners"

for folder in "${POSITIONS}"/*/; do
    folder_name=$(basename "$folder")
    echo "Annotating folder ${folder_name}..."
    python -m src.annotate_corners --folder "$folder" --out "${CORNERS}/${folder_name}"
done

echo "All folders annotated."
