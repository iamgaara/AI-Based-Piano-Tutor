@echo off
title Piano AI - Quantitative Benchmark
set "ROOT_DIR=%~dp0"
set "TEMP=%ROOT_DIR%.temp"
set "TMP=%ROOT_DIR%.temp"
set "YOLO_CONFIG_DIR=%ROOT_DIR%.cache\yolo"

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"

echo Running CPU benchmark on demo footage...
"%ROOT_DIR%.venv\Scripts\python.exe" "%ROOT_DIR%Experiments\piano_detection\test_baseline.py"
echo.
echo Benchmark complete! Output snapshots saved in Experiments\piano_detection\output_baseline\
pause
