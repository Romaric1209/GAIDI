from glob import glob
from tqdm import tqdm
import os

from extract_mouth_frames import extract_mouth_from_video
from extract_audio_features import extract_mfcc_from_video

def preprocess_all(input_dir, output_dir):
    video_files = glob(f"{input_dir}/*.mp4")
    for video_path in tqdm(video_files):
        name = os.path.splitext(os.path.basename(video_path))[0]
        video_out = os.path.join(output_dir, name)
        extract_mouth_from_video(video_path, os.path.join(video_out, "frames"))
        extract_mfcc_from_video(video_path, os.path.join(video_out, "audio.npy"))


