@echo off
title Piano AI - Live Webcam & Mic App
echo Starting AI-Piano-Tutor with live webcam (Camera 0) and microphone...
set TEMP=D:\Final year project\.temp
set TMP=D:\Final year project\.temp
set YOLO_CONFIG_DIR=D:\Final year project\.cache\yolo
set PYTHONUNBUFFERED=1

cd /d "D:\Final year project\Research\AI-Piano-Tutor"
"D:\Final year project\.venv\Scripts\python.exe" main.py
pause
