import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add research repo to path so we can import its modules without modifying them
RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Research", "AI-Piano-Tutor"))
sys.path.insert(0, RESEARCH_DIR)

from ultralytics import YOLO
from vision.constants import white_notes, black_notes, black_key_indices, note_colors

def evaluate_baseline_on_video(video_path, output_dir, max_frames=60):
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(RESEARCH_DIR, "piano_seg_yolov8", "runs", "segment", "train2", "weights", "best.pt")
    
    print(f"Loading YOLO segmentation model from: {weights_path}")
    model = YOLO(weights_path)
    model.names[0] = "piano"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height} @ {fps:.1f} FPS, {total_frames} total frames")
    
    frame_idx = 0
    inference_times = []
    line_counts = []
    keys_matched_count = 0
    
    line_buffer = []
    buffer_size = 10
    merge_thresh = 10
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        # The author's repo flips the webcam frame: frame = cv2.flip(frame, -1)
        # In the demo video, check if it's already right-side-up or needs flip
        results = model(frame, conf=0.8, verbose=False)[0]
        t1 = time.time()
        inference_times.append((t1 - t0) * 1000)
        
        has_mask = results.masks is not None and len(results.masks.data) > 0
        num_lines = 0
        is_52_matched = False
        
        if has_mask:
            yolo_mask = results.masks.data[0].cpu().numpy()
            yolo_mask = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))
            yolo_mask = (yolo_mask * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(yolo_mask, 128, 255, cv2.THRESH_BINARY)
            x, y, w, h = cv2.boundingRect(binary_mask)
            
            crop_margin = 10
            y1 = max(y - crop_margin, 0)
            y2 = min(y + h + crop_margin, frame.shape[0])
            x1 = max(x - crop_margin, 0)
            x2 = min(x + w + crop_margin, frame.shape[1])
            
            piano_crop = frame[y1:y2, x1:x2]
            piano_gray = cv2.cvtColor(piano_crop, cv2.COLOR_BGR2GRAY)
            
            # Key line extraction (from keyboard_ui.py)
            h_crop = piano_gray.shape[0]
            white_keys = piano_gray[int(h_crop * 0.7):]
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            equalized_white_keys = clahe.apply(white_keys)
            white_key_edges = cv2.Canny(equalized_white_keys, 50, 150, apertureSize=3)
            kernel = np.ones((3, 3), np.uint8)
            closed_edge = cv2.morphologyEx(white_key_edges, cv2.MORPH_CLOSE, kernel)
            
            lines = cv2.HoughLinesP(closed_edge, rho=1, theta=np.pi / 180, threshold=10, minLineLength=20, maxLineGap=10)
            vertical_lines = []
            if lines is not None:
                for line in lines:
                    coords = line[0] if (hasattr(line, '__len__') and len(line) == 1 and hasattr(line[0], '__len__')) else line
                    if hasattr(coords, '__len__') and len(coords) == 4:
                        lx1, ly1, lx2, ly2 = coords
                        angle = np.arctan2((ly2 - ly1), (lx2 - lx1)) * 180 / np.pi
                        if abs(angle) > 80:
                            vertical_lines.append((lx1, ly1, lx2, ly2))
            
            frame_x_centers = sorted([int((lx1 + lx2) / 2) for (lx1, ly1, lx2, ly2) in vertical_lines])
            line_buffer.append(frame_x_centers)
            if len(line_buffer) > buffer_size:
                line_buffer.pop(0)
                
            merged_xs = []
            if len(line_buffer) == buffer_size:
                all_xs = sorted([x for f_xs in line_buffer for x in f_xs])
                cluster = []
                for xs in all_xs:
                    if not cluster or abs(xs - cluster[-1]) <= merge_thresh:
                        cluster.append(xs)
                    else:
                        merged_xs.append(int(np.mean(cluster)))
                        cluster = [xs]
                if cluster:
                    merged_xs.append(int(np.mean(cluster)))
                    
            num_lines = len(merged_xs)
            line_counts.append(num_lines)
            if num_lines - 1 == len(white_notes):
                is_52_matched = True
                keys_matched_count += 1
                
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx:02d}: Mask detected={has_mask}, Lines detected={num_lines}, Exact 52-Key Match={is_52_matched}, YOLO Latency={inference_times[-1]:.1f}ms")
            # Save annotated snapshot
            annotated = results.plot(labels=False)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:02d}_seg.jpg"), annotated)
            
        frame_idx += 1
        
    cap.release()
    avg_latency = np.mean(inference_times)
    fps_estimate = 1000.0 / avg_latency if avg_latency > 0 else 0
    print("\n--- BASELINE EVALUATION REPORT ---")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Avg YOLOv8 Segmentation Latency (CPU): {avg_latency:.1f} ms (~{fps_estimate:.1f} FPS)")
    print(f"Frames with exact 52-Key Match (out of buffered): {keys_matched_count}/{max(1, frame_idx - buffer_size)}")
    if line_counts:
        print(f"Hough line counts range: min={min(line_counts)}, max={max(line_counts)}, mean={np.mean(line_counts):.1f} (target: 53 lines for 52 keys)")
        
if __name__ == "__main__":
    demo_video = os.path.join(RESEARCH_DIR, "Demo", "first_bar_single_notes.mp4")
    out_dir = os.path.join(os.path.dirname(__file__), "output_baseline")
    evaluate_baseline_on_video(demo_video, out_dir, max_frames=50)
