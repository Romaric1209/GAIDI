import os
import random
import subprocess

def create_fake_avsamples(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    for file in video_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        manipulation_type = random.choice(['shift', 'swap', 'mute'])

        if manipulation_type == 'shift':
            # Shift audio by a random amount between -1.0 and 1.0 seconds
            offset = round(random.uniform(-1.0, 1.0), 2)
            cmd = [
                'ffmpeg', '-y',
                '-itsoffset', str(offset), '-i', input_path,
                '-i', input_path,
                '-map', '1:v:0', '-map', '0:a:0',
                '-c:v', 'copy', '-shortest',
                output_path
            ]

        elif manipulation_type == 'swap':
            # Pick a random other video to use its audio
            candidates = [f for f in video_files if f != file]
            if not candidates:
                continue  # skip if no swap target
            swap_file = random.choice(candidates)
            swap_path = os.path.join(input_dir, swap_file)
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path, '-i', swap_path,
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'copy', '-shortest',
                output_path
            ]

        elif manipulation_type == 'mute':
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-an', output_path
            ]

        print(f"Creating {manipulation_type} fake: {file}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Example usage:
# create_fake_avsamples('/content/original_videos', '/content/fake_videos')


