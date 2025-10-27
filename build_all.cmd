@echo off
REM Configuration 
set "POSITIONS=input_imgs"
set "CORNERS=data\corners"
set "DATASET=data\dataset"

REM Loop through each subfolder
for /D %%F in ("%POSITIONS%\*") do (
  REM %%~nF
  if exist "%%F\fen_list.csv" (
    echo Building dataset for folder %%~nF...
    python -m src.build_dataset ^
      --folder "%%F" ^
      --corners-dir "%CORNERS%\%%~nF" ^
      --fen-file "%%F\fen_list.csv" ^
      --dataset-root "%DATASET%" ^
      --img-size 96
  ) else (
    echo Skipping %%F (no fen_list.csv found)
  )
)

echo All folders processed.
pause
