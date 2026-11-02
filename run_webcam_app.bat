@echo off
title Piano AI - Live Webcam & Mic App
set "ROOT_DIR=%~dp0"
set "TEMP=%ROOT_DIR%.temp"
set "TMP=%ROOT_DIR%.temp"
set "YOLO_CONFIG_DIR=%ROOT_DIR%.cache\yolo"
set PYTHONUNBUFFERED=1

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"

echo Starting AI-Piano-Tutor with live webcam (Camera 0) and microphone...
cd /d "%ROOT_DIR%Research\AI-Piano-Tutor"
"%ROOT_DIR%.venv\Scripts\python.exe" main.py %*
pause
