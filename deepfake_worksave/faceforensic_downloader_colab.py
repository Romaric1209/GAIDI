import os
import json
import urllib.request
from tqdm import tqdm
import random

# === User Config ===
OUTPUT_PATH = '/content/drive/MyDrive/Colab_Notebooks/GAIDI/Deepfake'
BASE_URL = 'http://kaldir.vc.in.tum.de/faceforensics/v3'
DATASETS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection'
}
DOWNLOAD_LOG = os.path.join(OUTPUT_PATH, 'downloaded_files_log.txt')
COMPRESSION = 'c23'
TYPE = 'videos'
NUM_REAL = 24
NUM_FAKE_PER_TYPE = 4  # 4 × 6 = 24 fake

random.seed(42)

# === Helper Functions ===
def fetch_json(url):
    with urllib.request.urlopen(url) as f:
        return json.loads(f.read().decode())

def get_already_downloaded():
    if not os.path.exists(DOWNLOAD_LOG):
        return set()
    with open(DOWNLOAD_LOG, 'r') as f:
        return set(line.strip() for line in f.readlines())

def mark_as_downloaded(entries):
    with open(DOWNLOAD_LOG, 'a') as f:
        for entry in entries:
            f.write(entry + '\n')

def download_file(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)

# === Main Logic ===
def main():
    downloaded = get_already_downloaded()
    print("Already downloaded:", len(downloaded))

    # Load file lists
    filelist_url = BASE_URL + '/misc/filelist.json'
    detection_filelist_url = BASE_URL + '/misc/deepfake_detection_filenames.json'
    file_pairs = fetch_json(filelist_url)
    detection_files = fetch_json(detection_filelist_url)

    to_download = []

    # === 28 Real Videos ===
    real_candidates = list(dict.fromkeys(f for pair in file_pairs for f in pair))
    real_remaining = [f for f in real_candidates if f"originals/{f}.mp4" not in downloaded]
    selected_real = random.sample(real_remaining, min(NUM_REAL, len(real_remaining)))
    to_download += [("originals", f, DATASETS['original']) for f in selected_real]

    # === 4 Fake Videos Per Type ===
    for key in DATASETS:
        if key == 'original':
            continue

        if key == 'DeepFakeDetection':
            # deepfake_detection_filenames.json uses 'deepfakes' as key
            candidates = [f for f in detection_files['DeepFakesDetection'] if f"manipulated/{key}/{f}.mp4" not in downloaded]
        else:
            # Create A_B and B_A combinations
            pair_candidates = [('_'.join(p), '_'.join(p[::-1])) for p in file_pairs]
            flattened = list(set(f for pair in pair_candidates for f in pair))
            candidates = [f for f in flattened if f"manipulated/{key}/{f}.mp4" not in downloaded]

        if len(candidates) == 0:
            print(f"⚠️  No more available fake videos for {key}")
            continue

        selected = random.sample(candidates, min(NUM_FAKE_PER_TYPE, len(candidates)))
        to_download += [("manipulated/" + key, f, DATASETS[key]) for f in selected]

    print(f"\n🔻 Downloading {len(to_download)} new videos...")

    for subfolder, filename, dataset_path in tqdm(to_download):
        fname = filename + '.mp4'
        url = f"{BASE_URL}/{dataset_path}/{COMPRESSION}/{TYPE}/{fname}"
        out_path = os.path.join(OUTPUT_PATH, subfolder, fname)
        try:
            download_file(url, out_path)
            mark_as_downloaded([f"{subfolder}/{fname}"])
        except Exception as e:
            print(f"❌ Failed: {fname} — {e}")

    print(f"\n✅ Done: {len(to_download)} new videos downloaded.")

if __name__ == "__main__":
    main()
