import os
import sys
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path

from extract_audio_features import extract_features_torchaudio
from extract_face_crop import extract_face_crop_from_coords  # Updated version using CSV X,Y

# Set up base directory
BASE_DIR = '/content/drive/MyDrive/Colab_Notebooks/GAIDI/Deepfake/AVSpeech'
sys.path.append(BASE_DIR)

CSV_PATH = f'{BASE_DIR}/avsp_test.csv'
VIDEO_DIR = f'{BASE_DIR}/videos/train_subset/original_videos'
OUTPUT_REAL = f'{BASE_DIR}/processed/real'
OUTPUT_FAKE = f'{BASE_DIR}/processed/fake'


def preprocess_from_csv(csv_path, video_dir, output_dir):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['YouTube ID', 'start segment', 'end segment', 'X coordinate', 'Y coordinate']

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_id = row['YouTube ID']
        start = row['start segment']
        end = row['end segment']
        x = row['X coordinate']
        y = row['Y coordinate']

        name = f"{vid_id}_{start:.2f}_{end:.2f}"
        video_out = os.path.join(output_dir, name)

        # Support multiple formats (mp4, webm, etc.)
        possible_matches = glob(os.path.join(video_dir, name + ".*"))
        if not possible_matches:
            # print(f"[WARNING] Video file not found for {name}")
            continue

        video_path = possible_matches[0]

        # Crop based on (X, Y) from CSV
        extract_face_crop_from_coords(video_path, os.path.join(video_out, "frames"), x, y)

        # Extract audio features
        audio_result = extract_features_torchaudio(video_path, os.path.join(video_out, "audio.npy"))
        if audio_result is None:
            # print(f"[INFO] Audio skipped for {video_path}")
            continue  


# MAIN EXECUTION
if __name__ == '__main__':
    preprocess_from_csv(CSV_PATH, VIDEO_DIR, OUTPUT_REAL)

