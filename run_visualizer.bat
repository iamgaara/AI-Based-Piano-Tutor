@echo off
title Piano AI - Interactive Demo Visualizer
set "ROOT_DIR=%~dp0"
set "TEMP=%ROOT_DIR%.temp"
set "TMP=%ROOT_DIR%.temp"
set "YOLO_CONFIG_DIR=%ROOT_DIR%.cache\yolo"
set PYTHONUNBUFFERED=1

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"

echo Starting interactive video visualizer...
"%ROOT_DIR%.venv\Scripts\python.exe" "%ROOT_DIR%Experiments\piano_detection\view_demo_live.py" %*
pause
