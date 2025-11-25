# tools/extract_frames_mp4_fix.py
import os, cv2, json
from tqdm import tqdm

VID_DIR = "data/MERL/videos"
OUT_BASE = "data/MERL/frames"
os.makedirs(OUT_BASE, exist_ok=True)
meta = {}

def canonical_video_id(filename):
    # remove extension
    base = os.path.splitext(filename)[0]
    # common pattern: "10_1_crop" -> return "10_1"
    if base.endswith("_crop"):
        return base[:-5]
    # also handle any trailing "-crop" or similar just in case
    if base.endswith("-crop"):
        return base[:-5]
    return base

for vf in sorted([f for f in os.listdir(VID_DIR) if f.lower().endswith(".mp4")]):
    vidpath = os.path.join(VID_DIR, vf)
    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        print("Cannot open", vidpath)
        continue
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    base = canonical_video_id(vf)
    outdir = os.path.join(OUT_BASE, base)
    os.makedirs(outdir, exist_ok=True)
    idx = 0
    pbar = tqdm(total=frame_count, desc=f"Extract {vf}")
    while True:
        ret, frame = cap.read()
        if not ret: break
        idx += 1
        fname = f"frame_{idx:06d}.jpg"
        cv2.imwrite(os.path.join(outdir, fname), frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    meta[base] = {"fps": fps, "frames": idx, "source_file": vf}
# write meta
os.makedirs("data/MERL", exist_ok=True)
with open("data/MERL/videos_meta.json","w") as fh:
    json.dump(meta, fh, indent=2)
print("Frame extraction finished. Videos meta saved to data/MERL/videos_meta.json")
