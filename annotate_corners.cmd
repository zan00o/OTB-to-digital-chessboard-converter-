@echo off
set "POSITIONS=input_imgs"
set "CORNERS=data\corners"

for /D %%F in ("%POSITIONS%\*") do (
  echo Annotating folder %%~nF...
  python -m src.annotate_corners --folder "%%F" --out "%CORNERS%\%%~nF"
)

echo All folders annotated.
pause