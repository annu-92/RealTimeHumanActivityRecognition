# tools/create_videos_meta.py
import os
import json
import cv2
from tqdm import tqdm

VIDEO_DIR = "data/MERL/videos"
OUT_FILE = "data/MERL/videos_meta.json"

meta = {}

for fname in tqdm(sorted(os.listdir(VIDEO_DIR)), desc="Processing videos"):
    if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    
    path = os.path.join(VIDEO_DIR, fname)
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"Cannot open {fname}")
        continue
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video ID should match folder names in frames directory
    vid_id = os.path.splitext(fname)[0]

    meta[vid_id] = {
        "frames": frame_count,
        "fps": fps,
        "height": h,
        "width": w
    }

    cap.release()

# Save metadata
with open(OUT_FILE, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved metadata to {OUT_FILE}")
print("Videos found:", len(meta))
