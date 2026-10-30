@echo off
title Piano AI - Interactive Demo Visualizer
echo Starting interactive video visualizer on Drive D...
set TEMP=D:\Final year project\.temp
set TMP=D:\Final year project\.temp
set YOLO_CONFIG_DIR=D:\Final year project\.cache\yolo

"D:\Final year project\.venv\Scripts\python.exe" "D:\Final year project\Experiments\piano_detection\view_demo_live.py"
pause
