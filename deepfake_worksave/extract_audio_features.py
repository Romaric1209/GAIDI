import torchaudio
import torchaudio.transforms as T
import torch
import os
import numpy as np

def extract_features_torchaudio(input_path, output_path, sr=16000, n_mels=40):
    try:
        waveform, orig_sr = torchaudio.load(input_path)
    except Exception as e:
        return None # skip the video instead of sending an error back

    if orig_sr != sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    db_transform = T.AmplitudeToDB(stype='power')
    mel_db = db_transform(mel_spec).squeeze(0).T

    mel_db = (mel_db - mel_db.mean()) / mel_db.std() # normalization

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, mel_db.numpy())

    return mel_db