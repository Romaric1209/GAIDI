import librosa
import numpy as np
from pathlib import Path
import os

def extract_mfcc_from_video(video_path, output_path, sr=16000, n_mfcc=13):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(video_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    np.save(output_path, mfcc.T)
