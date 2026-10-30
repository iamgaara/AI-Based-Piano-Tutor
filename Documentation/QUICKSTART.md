# Project Quickstart & Testing Guide

Everything is saved and configured on **Drive D:** with zero impact on Drive C.

---

## 🚀 Quick Launch (Double-Click in Windows Explorer)

You can launch tests tomorrow morning directly by double-clicking any of these `.bat` scripts in `D:\Final year project\`:

1. **`run_visualizer.bat`**  
   Opens interactive OpenCV windows displaying YOLOv8 segmentation and Hough line key detection over real piano footage.
   - `[SPACE]` to pause/resume
   - `[ESC]` to exit

2. **`run_benchmark.bat`**  
   Runs the 50-frame quantitative CPU benchmark and logs per-frame latency, FPS, and line counts.
   - Snapshots saved to: `Experiments/piano_detection/output_baseline/`

3. **`run_webcam_app.bat`**  
   Launches the original full application using your physical webcam (Camera 0) and microphone (Device 1).
   - Press `l` to lock keyboard layout once piano is visible.

---

## 📁 Directory Overview

- **`Research/AI-Piano-Tutor/`**: Original cloned repository (unmodified research baseline).
- **`Experiments/piano_detection/`**:
  - `test_baseline.py`: Non-interactive quantitative benchmark script.
  - `view_demo_live.py`: Interactive real-time visualizer script.
  - `output_baseline/`: Saved segmentation snapshot images.
- **`Documentation/`**:
  - `research_baseline_report.md`: In-depth analysis of baseline results, failure modes, and next steps.
  - `QUICKSTART.md`: This quick reference guide.
- **`.venv/`**: Python 3.10 virtual environment containing all dependencies (CPU PyTorch, YOLOv8, OpenCV, MediaPipe, Music21).
- **`.cache/` & `.temp/`**: Custom cache directories protecting Drive C.

---

## 💻 Manual Terminal Commands

If you prefer using PowerShell:
```powershell
# Interactive visualizer:
& "D:\Final year project\.venv\Scripts\python.exe" "D:\Final year project\Experiments\piano_detection\view_demo_live.py"

# Benchmark:
& "D:\Final year project\.venv\Scripts\python.exe" "D:\Final year project\Experiments\piano_detection\test_baseline.py"

# Full webcam app:
cd "D:\Final year project\Research\AI-Piano-Tutor"
& "D:\Final year project\.venv\Scripts\python.exe" main.py
```
