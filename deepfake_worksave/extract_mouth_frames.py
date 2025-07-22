import cv2
import os
import mediapipe as mp
from pathlib import Path

def extract_mouth_from_video(video_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            mouth_lm = [lm[i] for i in range(61, 88)]
            h, w, _ = frame.shape
            xs = [int(p.x * w) for p in mouth_lm]
            ys = [int(p.y * h) for p in mouth_lm]
            x1, x2 = max(min(xs)-10, 0), min(max(xs)+10, w)
            y1, y2 = max(min(ys)-10, 0), min(max(ys)+10, h)
            mouth_crop = frame[y1:y2, x1:x2]
            out_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(out_path, mouth_crop)
        frame_idx += 1
