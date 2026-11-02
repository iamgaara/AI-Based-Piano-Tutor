# vision/keyboard_ui.py

import os
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import mediapipe as mp
import logging
from vision.constants import white_notes, black_notes, black_key_indices, note_colors

class KeyboardUI:
    def __init__(self, video_source=0, flip_frame=None, loop=True, debug=False):
        print("✅ Keyboard UI initialized")
        self.video_source = video_source
        self.loop = loop
        self.debug = debug
        is_cam = isinstance(video_source, int) or (isinstance(video_source, str) and str(video_source).isdigit())
        self.is_cam = is_cam

        if flip_frame is None:
            self.flip_frame = True if is_cam else False
        else:
            self.flip_frame = flip_frame

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "..", "piano_seg_yolov8", "runs", "segment", "train2", "weights", "best.pt")
        if not os.path.exists(weights_path):
            weights_path = "piano_seg_yolov8/runs/segment/train2/weights/best.pt"
        self.model = YOLO(weights_path)
        LOGGER.setLevel(logging.CRITICAL)

        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        cam_idx = int(video_source) if is_cam else video_source
        self.cap = cv2.VideoCapture(cam_idx)
        if is_cam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.locked = False
        self.locked_merged_xs = None
        self.locked_crop_coords = None
        self.current_note = None
        self.last_flash_time = 0
        self.crop_margin = 10
        self.buffer_size = 10

        self.notes = white_notes
        self.black_notes = black_notes
        self.black_indices = black_key_indices 
        self.note_colors = note_colors

        self.font = ImageFont.truetype("arialbd.ttf", size=10)  # "bd" = bold
        self.last_clean_time = time.time()
        self.clean_reference = None
        self.line_buffer = []

        self.circle_radius = 10
        self.merge_thresh = 10

        self._measure_image = None
        self._show_measure = False

    # def flash_note(self, note_name):
    #     self.current_note = note_name
    #     self.last_flash_time = time.time()

    def flash_note(self, note_dict):
        # if isinstance(note_names, str):  # If it's a single note, wrap in list
        #     note_names = [note_names]
        # note_dict = {"treble": [...], "bass": [...]}
        print(f"⚡ Flashing note(s): {note_dict}")  # Add this line
        self.current_note = note_dict
        self.last_flash_time = time.time()

    def match_brightness(self, src, ref):
        # src and ref should be grayscale images

        src_mean = np.mean(src)
        ref_mean = np.mean(ref)

        # Avoid division by zero
        if ref_mean == 0:
            return ref.copy()

        scale = src_mean / ref_mean
        adjusted = np.clip(ref.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return adjusted

    def draw_note_circle(self, draw, center_x, center_y, note, is_target=False):
        letter = ''.join(filter(str.isalpha, note))
        base_color = self.note_colors.get(letter, (128, 128, 128))
        if is_target and isinstance(self.current_note, dict):
            if note in self.current_note.get("treble", []):
                color = base_color  # Light sky blue (calm but clear)
                outline_color = (255, 255, 0)
                outline_radius_margin = 5
            elif note in self.current_note.get("bass", []):
                color = base_color  # Soft mint green (friendly, readable)
                outline_color = (255, 165, 60)
                outline_radius_margin = 5
            else:
                color = (255, 255, 255)  # Fallback
        else:
            color = base_color
            outline_color = (0,0,0)  # Black outline for non-target notes
            outline_radius_margin = 2

        radius = 10 if is_target else self.circle_radius
        outline_radius = radius + outline_radius_margin
        draw.ellipse(
            [(center_x - outline_radius, center_y - outline_radius),
            (center_x + outline_radius, center_y + outline_radius)],
            fill=outline_color
        )

        # Inner colored circle
        draw.ellipse(
            [(center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius)],
            fill=color
        )

        try:
            bbox = draw.textbbox((0, 0), note, font=self.font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            w, h = self.font.getsize(note)

        draw.text((center_x - w/2, center_y - h/2), note, fill=(0, 0, 0), font=self.font)


    def toggle_lock(self, current_xs, current_crop_coords):
        self.locked = not self.locked
        if self.locked:
            self.locked_merged_xs = current_xs.copy()
            self.locked_crop_coords = current_crop_coords
            print("🔒 Locked keyboard layout.")
        else:
            self.locked_merged_xs = None
            self.locked_crop_coords = None
            print("🔓 Unlocked keyboard layout.")

    def show_measure_image(self, image):
        self._measure_image = image
        self._show_measure = True

    def show(self):
        print("📷 Showing UI...")
        while True:
            if self._show_measure and self._measure_image is not None:
                cv2.imshow("Sheet Music", self._measure_image)
                cv2.moveWindow("Sheet Music", 100, 100)

            ret, frame = self.cap.read()
            if not ret:
                if self.loop and not self.is_cam:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            if self.flip_frame:
                frame = cv2.flip(frame, -1)  # Flip upside down
            frame_bgr = frame.copy()
            piano_bgr = None
            xs_to_use = []

            # FAST PATH: If locked, bypass heavy AI (YOLO, MediaPipe, HoughLines) for high FPS
            if self.locked and self.locked_crop_coords and self.locked_merged_xs:
                x, y, w, h, x2, y2, w2, h2 = self.locked_crop_coords
                y1 = max(y - self.crop_margin, 0)
                y2_outer = min(y + h + self.crop_margin, frame_bgr.shape[0])
                x1 = max(x - self.crop_margin, 0)
                x2_outer = min(x + w + self.crop_margin, frame_bgr.shape[1])
                piano_YOLO_bgr = frame_bgr[y1:y2_outer, x1:x2_outer]
                if piano_YOLO_bgr.shape[0] >= y2 + h2 and piano_YOLO_bgr.shape[1] >= x2 + w2:
                    piano_bgr = piano_YOLO_bgr[y2:y2 + h2, x2:x2 + w2]
                else:
                    piano_bgr = piano_YOLO_bgr
                xs_to_use = self.locked_merged_xs
            else:
                # FULL DETECTION PATH (Runs only during setup before 'L' is pressed)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_hands = self.hands.process(frame_rgb)

                # YOLOv8 segmentation
                self.model.names[0] = "piano"
                results = self.model(frame, conf=0.8, verbose=False)[0]

                if not self.debug:
                    annotated = results.plot(labels=False)
                    disp_h = 540
                    disp_w = max(1, int(annotated.shape[1] * (disp_h / annotated.shape[0])))
                    cv2.imshow("Piano Setup (Press 'L' to Lock)", cv2.resize(annotated, (disp_w, disp_h)))
                else:
                    annotated = results.plot(labels=False)
                    cv2.imshow("Original frame and YOLO mask", annotated)

                h_img, w_img, _ = frame.shape
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if results_hands.multi_hand_landmarks is None:
                    if self.clean_reference is None or time.time() - self.last_clean_time > 2:
                        self.clean_reference = frame_gray.copy()
                        self.last_clean_time = time.time()
                else:
                    if self.clean_reference is not None:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            xs_hand = [int(landmark.x * w_img) for landmark in hand_landmarks.landmark]
                            ys_hand = [int(landmark.y * h_img) for landmark in hand_landmarks.landmark]
                            x_min, x_max = max(min(xs_hand) - 20, 0), min(max(xs_hand) + 20, w_img)
                            y_min, y_max = max(min(ys_hand) - 20, 0), min(max(ys_hand) + 20, h_img)
                            clean_reference_matched = self.match_brightness(frame_gray, self.clean_reference)
                            frame_gray[y_min:y_max, x_min:x_max] = clean_reference_matched[y_min:y_max, x_min:x_max]

                if results.masks is not None and len(results.masks.data) > 0:
                    yolo_mask = results.masks.data[0].cpu().numpy()
                    yolo_mask = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))
                    yolo_mask = (yolo_mask * 255).astype(np.uint8)
                    _, binary_mask = cv2.threshold(yolo_mask, 128, 255, cv2.THRESH_BINARY)
                    x, y, w, h = cv2.boundingRect(binary_mask)

                    y1 = max(y - self.crop_margin, 0)
                    y2_outer = min(y + h + self.crop_margin, frame.shape[0])
                    x1 = max(x - self.crop_margin, 0)
                    x2_outer = min(x + w + self.crop_margin, frame.shape[1])

                    piano_YOLO = frame_gray[y1:y2_outer, x1:x2_outer]
                    piano_YOLO_bgr = frame_bgr[y1:y2_outer, x1:x2_outer]

                    _, bright_mask = cv2.threshold(piano_YOLO, 180, 255, cv2.THRESH_BINARY)
                    kernel = np.ones((5, 5), np.uint8)
                    closed_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        x2, y2, w2, h2 = cv2.boundingRect(largest)
                        piano = piano_YOLO[y2:y2 + h2, x2:x2 + w2]
                        piano_bgr = piano_YOLO_bgr[y2:y2 + h2, x2:x2 + w2]
                    else:
                        piano = piano_YOLO
                        piano_bgr = piano_YOLO_bgr
                        x2, y2, w2, h2 = 0, 0, piano.shape[1], piano.shape[0]

                    if self.debug:
                        cv2.imshow("Refined Keyboard Crop", piano)

                    h_crop = piano.shape[0]
                    white_keys = piano[int(h_crop * 0.7):]
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    equalized_white_keys = clahe.apply(white_keys)
                    white_key_edges = cv2.Canny(equalized_white_keys, 50, 150, apertureSize=3)
                    kernel_close = np.ones((3, 3), np.uint8)
                    closed_edge = cv2.morphologyEx(white_key_edges, cv2.MORPH_CLOSE, kernel_close)

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

                    frame_x_centers = [int((x_1 + x_2) / 2) for (x_1, y_1, x_2, y_2) in vertical_lines]
                    frame_x_centers.sort()
                    merged_xs = []
                    self.line_buffer.append(frame_x_centers)
                    if len(self.line_buffer) > self.buffer_size:
                        self.line_buffer.pop(0)

                    if len(self.line_buffer) == self.buffer_size:
                        all_xs = [x_val for frame_xs in self.line_buffer for x_val in frame_xs]
                        all_xs.sort()
                        cluster = []
                        for xs_val in all_xs:
                            if not cluster or abs(xs_val - cluster[-1]) <= self.merge_thresh:
                                cluster.append(xs_val)
                            else:
                                merged_xs.append(int(np.mean(cluster)))
                                cluster = [xs_val]
                        if cluster:
                            merged_xs.append(int(np.mean(cluster)))

                    xs_to_use = merged_xs
                    if self.debug:
                        merged_line_img = piano.copy()
                        for x_line in xs_to_use:
                            cv2.line(merged_line_img, (x_line, 0), (x_line, merged_line_img.shape[0]), (0, 255, 0), 2)
                        cv2.imshow("Piano lines", merged_line_img)
                
            if 'piano_bgr' in locals() and piano_bgr is not None:
                cropped_copy = piano_bgr.copy()
                y_pos_white = int(cropped_copy.shape[0] * 0.9)
                y_pos_black = int(cropped_copy.shape[0] * 0.7)

                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_copy, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(cropped_pil)

                # Draw white keys
                if len(xs_to_use) - 1 == len(self.notes):
                    for i in range(len(xs_to_use) - 1):
                        note = self.notes[i]
                        mid_x = int((xs_to_use[i] + xs_to_use[i + 1]) / 2)
                        is_target = (
                            self.locked and isinstance(self.current_note, dict) and
                            (note in self.current_note.get("treble", []) or note in self.current_note.get("bass", []))
                        )
                        self.draw_note_circle(draw, mid_x, y_pos_white, note, is_target)

                # Draw black keys
                for idx, note in zip(self.black_indices, self.black_notes):
                    if idx < len(xs_to_use) - 1:
                        mid_x = int(xs_to_use[idx])
                        is_target = (
                            self.locked and isinstance(self.current_note, dict) and
                            (note in self.current_note.get("treble", []) or note in self.current_note.get("bass", []))
                        )
                        self.draw_note_circle(draw, mid_x, y_pos_black, note, is_target)

                # Convert and show
                frame_show = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                cv2.imshow("Piano Trainer", frame_show)
            else:
                display_frame = frame_bgr.copy()
                cv2.putText(display_frame, "Searching for piano... Point camera at keyboard", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Piano Trainer", cv2.resize(display_frame, (960, 540)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'):
                if not self.locked and 'x2' in locals() and merged_xs:
                    self.locked = True
                    self.locked_merged_xs = merged_xs.copy()
                    self.locked_crop_coords = (x, y, w, h, x2, y2, w2, h2)
                    print("🔒 Locked piano layout! Heavy AI bypassed for smooth 60+ FPS.")
                    try:
                        cv2.destroyWindow("Piano Setup (Press 'L' to Lock)")
                    except Exception:
                        pass
                elif self.locked:
                    self.locked = False
                    self.locked_merged_xs = None
                    self.locked_crop_coords = None
                    print("🔓 Unlocked piano layout. Resuming detection.")
                else:
                    print("Lock skipped — crop or keys not ready yet.")

            if key == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

