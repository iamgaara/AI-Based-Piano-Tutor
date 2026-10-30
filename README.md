# AI-Based Piano Tutor: Real-Time Keyboard Detection & Sight-Reading Assistant

> A computer vision and audio-guided piano tutoring system for beginner pianists. Maps sheet music notes to real piano keys in real time using webcam feed, neural keyboard segmentation, and audio note verification.

---

## Attribution & Academic Acknowledgements

This project builds upon the foundational research, vision pipeline, and architecture developed by **Aaron Mok**:

- **Original Author**: [Aaron Mok (@Aaron-Mok)](https://github.com/Aaron-Mok)
- **Original Repository**: [`Aaron-Mok/AI-Piano-Tutor`](https://github.com/Aaron-Mok/AI-Piano-Tutor)
- **Author's Technical Write-up**: [AI Piano Tutor – Real-Time Key and Note Detection](https://aaron-mok.github.io/blog/pianokeydetection)
- **Original License**: [GNU General Public License v3.0 (GPL-3.0)](https://github.com/Aaron-Mok/AI-Piano-Tutor/blob/main/LICENSE)
- **Dataset**: [Roboflow Universe Piano Segmentation Dataset](https://universe.roboflow.com/aaronmok/piano-k3vfc-tyaiw/dataset/1) (MIT License)

We maintain the original repository as an unmodified submodule and baseline in `Research/AI-Piano-Tutor/` for empirical benchmarking and comparative analysis.

---

## Repository Architecture

```
AI-Based-Piano-Tutor/
│
├── Research/
│   └── AI-Piano-Tutor/         # Aaron Mok's original cloned repository (baseline)
│
├── Experiments/
│   ├── piano_detection/        # Baseline benchmark and live demo visualizer
│   │   ├── test_baseline.py    # Automated CPU latency & key registration benchmark
│   │   └── view_demo_live.py   # Real-time visualizer on demo footage
│   ├── key_localization/       # Perspective homography & key subdivision experiments
│   └── model_comparison/       # Alternative lightweight model architectures
│
├── Dataset/                    # Custom keyboard and keypress annotations
├── Our_Implementation/         # Enhanced pipeline (Perspective-corrected key localization)
└── Documentation/
    ├── research_baseline_report.md  # Detailed benchmark metrics and failure analysis
    └── QUICKSTART.md           # Quick testing guide
```

---

## Baseline Reproduction & Benchmark Findings

We conducted an empirical reproduction and CPU benchmark using the author's pretrained weights on 1080p demo footage (`Demo/first_bar_single_notes.mp4`):

| Metric | Measured Baseline Performance | Status |
|---|---|---|
| **YOLOv8 Keyboard Segmentation** | 77.8 ms – 117.2 ms (~10.4 FPS steady-state on CPU) |  Highly Robust (100% mask detection) |
| **Hough Line Key Detection** | Detected 20–41 lines (Required: 53 lines for 52 white keys) | ⚠️ Sensitive to lighting & perspective |
| **Exact 52-Key Match Rate** | 0.0% (strict `len(lines) == 53` threshold) |  Critical failure mode under occlusion |
| **Audio Note Recognition** | FFT peak-picking with 3s background subtraction |  Functional for clean single notes |

For the full benchmark analysis, see [`Documentation/research_baseline_report.md`](Documentation/research_baseline_report.md).

---

## Key Improvements in Our Implementation

1. **4-Corner Homography & Perspective Warping**:
   Instead of simple bounding box cropping, we detect the 4 keyboard corners and warp the angled trapezoid into an orthogonal bird's-eye rectangle.
2. **Ratio-Based Geometric Key Subdivision**:
   Instead of fragile Hough edge counting, we use exact physical piano octave key ratios ($7:5$ white-to-black key ratio), supporting 49, 61, and 88-key layouts.
3. **Harmonic Product Spectrum (HPS) Audio Tracking**:
   Eliminates octave confusion and enhances polyphonic chord tracking.

---

## ⚡ Quickstart

### 1. Prerequisites & Virtual Environment Setup
```powershell
# Create isolated environment
py -3.10 -m venv .venv
& .venv\Scripts\Activate.ps1

# Install CPU PyTorch and dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install opencv-python ultralytics mediapipe music21 sounddevice scipy matplotlib
```

### 2. One-Click Testing (Windows)
Double-click any launcher in the project root:
- **`run_visualizer.bat`**: Interactive real-time visualizer displaying YOLO segmentation and key line overlays.
- **`run_benchmark.bat`**: Runs the 50-frame quantitative CPU benchmark.
- **`run_webcam_app.bat`**: Runs the live tutor with webcam (Camera 0) and microphone.

### 3. Manual Execution
```powershell
# Run interactive visualizer on footage
python Experiments/piano_detection/view_demo_live.py

# Run baseline benchmark
python Experiments/piano_detection/test_baseline.py

# Run live tutor
cd Research/AI-Piano-Tutor
python main.py
```

---

## 📄 License & Compliance

- Modifications, benchmarks, and new implementations in `Experiments/`, `Our_Implementation/`, and `Documentation/` are released under the **GNU General Public License v3.0 (GPL-3.0)** to maintain compatibility with the baseline.
- Pretrained dataset from Roboflow Universe is licensed under **MIT**.
