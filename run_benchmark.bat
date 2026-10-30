@echo off
title Piano AI - Quantitative Benchmark
echo Running CPU benchmark on demo footage...
set TEMP=D:\Final year project\.temp
set TMP=D:\Final year project\.temp
set YOLO_CONFIG_DIR=D:\Final year project\.cache\yolo

"D:\Final year project\.venv\Scripts\python.exe" "D:\Final year project\Experiments\piano_detection\test_baseline.py"
echo.
echo Benchmark complete! Output snapshots saved in Experiments\piano_detection\output_baseline\
pause
