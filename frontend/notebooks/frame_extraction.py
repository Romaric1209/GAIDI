import cv2
import torch
from torchvision import transforms
from PIL import Image

def extract_frames(video_path, wanted_fps = 10, resize=(224,224)):
    cap =cv2.VideoCapture(video_path)
    video_fps=cap.get(cv2.CAP_PROP_FPS)
    frames_interval = max(1,int(video_fps/wanted_fps))

    frames =[]
    count = 0

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    while True:
        ret,frame = cap.read()
        if not ret:
            break
        if count % frames_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(transform(image))
        count += 1

    cap.release()
    return torch.stack(frames)