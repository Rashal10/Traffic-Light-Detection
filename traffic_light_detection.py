import cv2
import numpy as np
import os
import time
from collections import deque

video_path = 0  
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 20

out = cv2.VideoWriter('output_annotated_improved.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

os.makedirs('screenshots', exist_ok=True)

# --- HSV color ranges (tuned) ---
red_lower1 = np.array([0, 120, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 120, 100])
red_upper2 = np.array([179, 255, 255])

yellow_lower = np.array([20, 120, 100])
yellow_upper = np.array([35, 255, 255])

green_lower = np.array([40, 100, 100])
green_upper = np.array([90, 255, 255])

# --- Tracking setup ---
tracking_buffer_size = 5  # how many frames to keep
stable_frames_required = 3  # confirm detection after seen in this many frames

tracked_objects = {
    'RED': deque(maxlen=tracking_buffer_size),
    'YELLOW': deque(maxlen=tracking_buffer_size),
    'GREEN': deque(maxlen=tracking_buffer_size)
}

def is_stable(center, centers_deque, max_dist=20):
    """Check if center is close to any center in history"""
    for c in centers_deque:
        dist = np.linalg.norm(np.array(center) - np.array(c))
        if dist < max_dist:
            return True
    return False

def detect_traffic_lights(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    detected_lights = []

    def find_lights(mask, color_name, color_bgr):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.5:
                continue

            # Solidity check
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = float(area) / hull_area
            if solidity < 0.85:
                continue

            # Ellipse fit check
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, orientation) = ellipse
                major_axis, minor_axis = axes
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                if eccentricity > 0.8:  # filter out very elongated shapes
                    continue

            detected_lights.append({
                'color': color_name,
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'color_bgr': color_bgr
            })

    find_lights(mask_red, "RED", (0, 0, 255))
    find_lights(mask_yellow, "YELLOW", (0, 255, 255))
    find_lights(mask_green, "GREEN", (0, 255, 0))

    return detected_lights

# --- Stats for accuracy report ---
label_counts = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
detection_counts = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
correct_counts = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or cannot read frame.")
        break

    lights = detect_traffic_lights(frame)

    confirmed_lights = []
    for light in lights:
        color = light['color']
        center = light['center']

        if is_stable(center, tracked_objects[color]):
            tracked_objects[color].append(center)
        else:
            tracked_objects[color].append(center)

        if len(tracked_objects[color]) >= stable_frames_required:
            confirmed_lights.append(light)

    for light in confirmed_lights:
        x, y, w, h = light['bbox']
        color_bgr = light['color_bgr']
        label = light['color']

        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    out.write(frame)

    if len(confirmed_lights) > 0:
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f'screenshots/detected_{timestamp}.jpg', frame)

    detected_colors = set(light['color'] for light in confirmed_lights)
    cv2.putText(frame, "Press r/y/g to label RED/YELLOW/GREEN presence, q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Detected: {','.join(detected_colors) if detected_colors else 'None'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Traffic Light Detection Improved", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Manual labeling by user - update counts first
    if key == ord('r'):
        label_counts['RED'] += 1
        if 'RED' in detected_colors:
            correct_counts['RED'] += 1
    elif key == ord('y'):
        label_counts['YELLOW'] += 1
        if 'YELLOW' in detected_colors:
            correct_counts['YELLOW'] += 1
    elif key == ord('g'):
        label_counts['GREEN'] += 1
        if 'GREEN' in detected_colors:
            correct_counts['GREEN'] += 1

    # Count detections ONLY on labeled frames (when key r/y/g pressed)
    if key in [ord('r'), ord('y'), ord('g')]:
        for color in ['RED', 'YELLOW', 'GREEN']:
            if color in detected_colors:
                detection_counts[color] += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n--- Detection Accuracy Report ---")
for color in ['RED', 'YELLOW', 'GREEN']:
    if label_counts[color] > 0:
        precision = correct_counts[color] / detection_counts[color] if detection_counts[color] > 0 else 0
        recall = correct_counts[color] / label_counts[color]
        print(f"{color}:")
        print(f"  Labeled Frames: {label_counts[color]}")
        print(f"  Detected Frames (on labeled frames): {detection_counts[color]}")
        print(f"  Correct Detections: {correct_counts[color]}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}\n")
    else:
        print(f"{color}: No labeled frames.\n")
