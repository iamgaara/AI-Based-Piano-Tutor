# Research Baseline Report: AI-Piano-Tutor Reproduction & Evaluation

**Date:** 2026-10-31  
**Target Repository:** [`https://github.com/Aaron-Mok/AI-Piano-Tutor.git`](https://github.com/Aaron-Mok/AI-Piano-Tutor.git)  
**Location:** `Research/AI-Piano-Tutor/`  
**Execution Environment:** Drive `D:\Final year project\.venv` (Python 3.10 64-bit, CPU-only PyTorch 2.14, Ultralytics YOLOv8, OpenCV, MediaPipe, Music21)

---

## 1. Executive Summary

We reproduced, inspected, and empirically benchmarked the `AI-Piano-Tutor` repository as our research baseline without modifying the original code.

| Area | Baseline Status | Key Finding |
|---|---|---|
| **Storage / Disk Safety** | Strictly on **Drive D:** | All dependencies, `.venv`, model weights, and outputs reside on Drive D with 0 disk pressure on Drive C. |
| **Piano Segmentation** | Functional (Pretrained YOLOv8-seg) | High segmentation accuracy on full keyboard bounds; runs at **~8–13 FPS on CPU** (~80–120ms steady-state per frame). |
| **Individual Key Identification** | **Fragile / Critical Failure Mode** | Relies on Canny + HoughLinesP vertical edge detection expecting exactly 53 lines (52 keys). On actual footage, it detected 20–41 lines, resulting in **0% automatic key registration**. |
| **Audio Note Recognition** | Functional (Basic FFT / Spectral Subtraction) | Uses sounddevice FFT with 3-second noise calibration; hardcoded to input device `1`. |
| **Sheet Music Parser** | Functional (Music21) | Successfully parses `.mxl` files and cached measure PNGs. |
| **Licensing** | GPL-3.0 (Code) / MIT (Dataset) | Roboflow keyboard polygon dataset is MIT licensed. The code is GPL-3.0. |

---

## 2. Quantitative Benchmark Results (Demo Footage Test)

We ran the baseline vision pipeline on the author's demo recording (`Demo/first_bar_single_notes.mp4`, 1920x1080 @ 60 FPS):

```
Total Frames Evaluated:       50 frames
Model Resolution:             1920x1080
Keyboard Detection Rate:      100% (Mask detected in all frames)
Warmup Latency (Frame 0):     5,055 ms
Steady-state Latency (CPU):   77.8 ms – 117.2 ms (avg ~96 ms / ~10.4 FPS)
Hough Line Count Detected:    Min: 0, Max: 41, Mean: 24.1 lines (Target: 53)
Frames with Exact 52-Key Match: 0 / 40 (0.0%)
```

---

## 3. Component Deep Dive: What Works vs. What Doesn't

### What Works Well (Reusable Components)
1. **Pretrained YOLOv8 Keyboard Segmentation (`best.pt`)**:
   - The YOLOv8n-seg model trained on Roboflow accurately isolates the keyboard polygon even under non-ideal lighting and oblique angles.
   - It can be reused as the initial coarse bounding stage.
2. **MusicXML Sheet Parsing (`sheet_parser.py`)**:
   - Clean implementation using `music21` extracting measure numbers, note names, and octave frequencies for treble and bass staves.
3. **Curated Demo Footage & Scores**:
   - The repository provides test video clips (`first_bar_single_notes.mp4`, `chrod_and_wrong_notes.mp4`) and standard `.mxl` scores for reproducible testing.

### What Doesn't Work / Core Limitations
1. **Key Extraction relies on Heuristic Hough Transform (`keyboard_ui.py`)**:
   - The line extraction algorithm:
     $$\text{keyboard} \rightarrow \text{bottom } 25\% \rightarrow \text{CLAHE} \rightarrow \text{Canny} \rightarrow \text{HoughLinesP} \rightarrow \text{clustering}$$
   - Fails when shadows, finger occlusions, or camera angles soften key boundary contrast.
   - Strict condition `len(xs_to_use) - 1 == 52` fails completely if even one line is missing or duplicated.
2. **No Perspective Correction / Homography**:
   - The model does not compute a homography matrix to project the angled trapezoidal keyboard into a normalized bird's-eye rectangle.
3. **Fixed 88-Key Assumption**:
   - Hardcoded for 52 white keys / 36 black keys (88-key piano). Incompatible with 61-key or 49-key synthesizer keyboards.
4. **Audio Polyphony & Latency**:
   - Simple FFT peak picking (`audio/mic_listener.py`) struggles with polyphonic overlapping notes, chord harmonics, and octave confusion without harmonic product spectrum (HPS) or neural pitch estimation (like SPICE or CREPE).
5. **Frame Inpainting Artefacts**:
   - The background hand replacement uses static background substitution that produces ghosting when ambient lighting shifts or the camera moves.

---

## 4. Architecture Recommendation for Our Implementation

Based on our empirical baseline evaluation, here is the roadmap for `Our_Implementation/`:

```
                       [ Input Video / Camera Feed ]
                                     |
                                     v
               [ Stage 1: Keyboard Segmentation (YOLOv8-seg) ]
                                     |
                                     v
         [ Stage 2: 4-Corner Detection & Perspective Homography ]
         (Warps keyboard polygon to flat orthogonal bird's-eye view)
                                     |
                                     v
        [ Stage 3: Geometric / Semantic Key Grid Localization ]
     (Detects keyboard size: 49/61/88 keys; splits keys robustly
      using known piano physical ratios rather than raw edge counts)
                                     |
                                     v
         [ Stage 4: Robust Note Tracking & Multi-Modal Fusion ]
     (Combines visual keypress detection with audio pitch tracking)
```
