import os
import sys
import cv2
import numpy as np

# Ensure research repo is accessible
RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Research", "AI-Piano-Tutor"))
sys.path.insert(0, RESEARCH_DIR)

from ultralytics import YOLO

def run_live_visualizer(video_path):
    weights_path = os.path.join(RESEARCH_DIR, "piano_seg_yolov8", "runs", "segment", "train2", "weights", "best.pt")
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)
    model.names[0] = "piano"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    print("\n--- Live Visualizer Running ---")
    print("Controls:")
    print("  [SPACE] : Pause / Resume")
    print("  [ESC]   : Exit")

    paused = False
    line_buffer = []
    buffer_size = 10
    merge_thresh = 10

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video. Looping back to start...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run YOLO segmentation
            results = model(frame, conf=0.8, verbose=False)[0]
            annotated = results.plot(labels=False)

            has_mask = results.masks is not None and len(results.masks.data) > 0
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

                piano_crop = frame[y1:y2, x1:x2].copy()
                piano_gray = cv2.cvtColor(piano_crop, cv2.COLOR_BGR2GRAY)

                # Key line extraction (CLAHE + Canny + HoughLines)
                h_crop = piano_gray.shape[0]
                white_keys = piano_gray[int(h_crop * 0.7):]
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                equalized_white_keys = clahe.apply(white_keys)
                white_key_edges = cv2.Canny(equalized_white_keys, 50, 150, apertureSize=3)
                kernel = np.ones((3, 3), np.uint8)
                closed_edge = cv2.morphologyEx(white_key_edges, cv2.MORPH_CLOSE, kernel)

                lines = cv2.HoughLinesP(closed_edge, rho=1, theta=np.pi / 180, threshold=10, minLineLength=20, maxLineGap=10)
                vertical_lines = []
                line_display = piano_crop.copy()

                if lines is not None:
                    for line in lines:
                        coords = line[0] if (hasattr(line, '__len__') and len(line) == 1 and hasattr(line[0], '__len__')) else line
                        if hasattr(coords, '__len__') and len(coords) == 4:
                            lx1, ly1, lx2, ly2 = coords
                            angle = np.arctan2((ly2 - ly1), (lx2 - lx1)) * 180 / np.pi
                            if abs(angle) > 80:
                                vertical_lines.append((lx1, ly1, lx2, ly2))
                                # Draw green line in crop coordinates
                                offset_y = int(h_crop * 0.7)
                                cv2.line(line_display, (lx1, ly1 + offset_y), (lx2, ly2 + offset_y), (0, 255, 0), 2)

                cv2.putText(line_display, f"Hough Lines: {len(vertical_lines)} (Target: 53)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Detected Piano Keys & Lines", cv2.resize(line_display, (960, int(line_display.shape[0] * 960 / line_display.shape[1]))))

            # Show annotated YOLO view
            display_annotated = cv2.resize(annotated, (960, 540))
            cv2.imshow("YOLOv8 Segmentation Overlay", display_annotated)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_video = os.path.join(RESEARCH_DIR, "Demo", "first_bar_single_notes.mp4")
    run_live_visualizer(demo_video)
