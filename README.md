# AI-Based Piano Tutor: Real-Time Keyboard Detection & Sight-Reading Assistant

> An augmented reality and audio-guided interactive piano tutor for beginner pianists. Maps sheet music notes directly to physical piano keys in real time using computer vision, neural keyboard segmentation, and audio pitch verification.

---

## 📌 Overview

Learning piano sight-reading is challenging for beginners: eyes constantly dart back and forth between the sheet music stand and physical keyboard. 

This system provides real-time interactive visual guidance:
1. **Parses Digital Sheet Music (`.mxl`):** Reads musical scores measure by measure.
2. **Locates Piano Keys via Computer Vision:** An overhead camera feeds into a YOLOv8 segmentation model and key-localization pipeline to map every white and black key.
3. **Augmented Reality Note Overlay:** Highlights the exact physical keys to strike with colored note targets on the live video feed.
4. **Real-Time Audio Verification:** Listens through the microphone using Fast Fourier Transform (FFT) pitch detection. Once you play the correct chord/note, the sheet music advances to the next measure automatically.

---

## 📂 Repository Structure

```
AI-Based-Piano-Tutor/
│
├── Research/
│   └── AI-Piano-Tutor/         # Core application & baseline
│       ├── main.py             # Main interactive tutor application
│       ├── vision/             # Camera capture & KeyboardUI rendering
│       ├── audio/              # Microphone listener & FFT pitch tracking
│       ├── piano_seg_yolov8/   # Trained YOLOv8 segmentation weights (best.pt)
│       ├── assets/scores/      # Sheet music MusicXML files
│       └── Demo/               # Sample demo footage for testing
│
├── Experiments/
│   ├── piano_detection/        # Baseline benchmark and live demo visualizer
│   │   ├── test_baseline.py    # Automated CPU latency & key registration benchmark
│   │   └── view_demo_live.py   # Real-time visualizer on video files
│   ├── key_localization/       # Perspective homography & key subdivision experiments
│   └── model_comparison/       # Lightweight neural model evaluations
│
├── Documentation/              # Research benchmarks & technical reports
├── requirements.txt            # Python dependencies
├── run_webcam_app.bat          # One-click launcher for live tutor
├── run_visualizer.bat          # One-click launcher for interactive visualizer
└── run_benchmark.bat           # One-click launcher for quantitative benchmark
```

---

## 🚀 Setup & Installation (New System Setup)

Follow these steps when cloning or forking this repository onto another system (CPU or GPU):

### 1. Clone the Repository
```powershell
git clone git@github.com:iamgaara/AI-Based-Piano-Tutor.git
cd AI-Based-Piano-Tutor
```

### 2. Create and Activate Virtual Environment
Requires **Python 3.10** or **3.11**:
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# (Or Command Prompt: .venv\Scripts\activate.bat)
```

### 3. Install Dependencies

#### 🎮 If your system has an NVIDIA GPU (CUDA):
Install PyTorch with CUDA acceleration first, then the remaining requirements:
```powershell
# Install PyTorch with CUDA 12.1 support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project requirements:
pip install -r requirements.txt
```

Verify your GPU is active:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

#### 💻 If your system is CPU-only:
```powershell
# Install CPU PyTorch:
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install project requirements:
pip install -r requirements.txt
```

---

## 🎹 How to Run

### Option A: One-Click Windows Launchers
Double-click any `.bat` script from the project folder:
- **`run_webcam_app.bat`**: Runs the live tutor with overhead webcam (Camera index 0) and microphone.
- **`run_visualizer.bat`**: Opens the live video visualizer to inspect YOLO segmentation and key lines.
- **`run_benchmark.bat`**: Executes the 50-frame quantitative CPU evaluation.

---

### Option B: Command-Line Execution

#### 1. Live Tutor with Overhead Webcam
```powershell
cd Research/AI-Piano-Tutor
python main.py
```
- Aim your camera at the keyboard until the overlay detects the piano.
- Press <kbd>L</kbd> to **lock the keyboard layout**. Once locked, heavy AI inference is bypassed, ensuring a silky-smooth **60+ FPS** experience during practice!

#### 2. Test with an Imported Video File
You can test without a live webcam by feeding any recorded `.mp4`:
```powershell
cd Research/AI-Piano-Tutor
python main.py --video "Demo/Left-hand-coordination-SnapYT.App.mp4" --no-mic
```

#### 3. Key Detection & Segmentation Visualizer
Inspect key boundary lines and segmentation on any video:
```powershell
python Experiments/piano_detection/view_demo_live.py --video "Research/AI-Piano-Tutor/Demo/Left-hand-coordination-SnapYT.App.mp4"
```

---

## ⌨️ In-App Controls

| Key | Action | Description |
|---|---|---|
| <kbd>L</kbd> | **Lock / Unlock Layout** | Locks the detected piano coordinates. Enables the fast-path bypass for 60+ FPS playback. Press again to unlock and reposition. |
| <kbd>SPACE</kbd> | **Pause / Resume** | Pauses or resumes playback in the visualizer. |
| <kbd>ESC</kbd> | **Exit** | Closes all open windows and releases camera/audio devices. |

---

## ⚙️ CLI Options & Flags

When running `main.py`, you can pass the following flags:
- `--video <path or index>`: Specify camera index (`0`, `1`) or path to a `.mp4` file.
- `--no-mic`: Disables microphone listening (ideal for visual-only testing).
- `--flip`: Forces frames to be flipped upside down (for inverted overhead camera mounts).
- `--no-flip`: Forces frames not to be flipped.
- `--debug`: Opens all intermediate computer vision debug windows (Canny edges, CLAHE, Hough lines, hands).

---

## 📜 Attribution & Academic Acknowledgements

This project builds upon the foundational research and baseline vision architecture by **Aaron Mok**:
- **Original Author**: [Aaron Mok (@Aaron-Mok)](https://github.com/Aaron-Mok)
- **Original Repository**: [`Aaron-Mok/AI-Piano-Tutor`](https://github.com/Aaron-Mok/AI-Piano-Tutor)
- **Dataset**: [Roboflow Universe Piano Segmentation Dataset](https://universe.roboflow.com/aaronmok/piano-k3vfc-tyaiw/dataset/1)
- **License**: GNU General Public License v3.0 (GPL-3.0)
