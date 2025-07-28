import cv2
import os
from pathlib import Path

def extract_face_crop_from_coords(video_path, output_dir, norm_x, norm_y, box_size=112):
    """
    Extracts a square face crop from the first frame using normalized (x, y) center.

    Parameters:
    - video_path: full path to the video
    - output_dir: directory to save cropped frame(s)
    - norm_x, norm_y: normalized coordinates from CSV (between 0 and 1)
    - box_size: size of the square crop (default: 112x112)
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Could not read first frame of {video_path}")
        return

    h, w, _ = frame.shape
    cx, cy = int(norm_x * w), int(norm_y * h)

    half_box = box_size // 2
    x1 = max(cx - half_box, 0)
    y1 = max(cy - half_box, 0)
    x2 = min(cx + half_box, w)
    y2 = min(cy + half_box, h)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"[WARNING] Empty crop in {video_path} at ({norm_x}, {norm_y})")
        return

    out_path = os.path.join(output_dir, f"frame_0000.jpg")
    cv2.imwrite(out_path, crop)

