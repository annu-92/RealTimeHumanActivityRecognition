# tools/check_seq_frames_for_bad_kps.py
import os, numpy as np

KP_DIR = "data/MERL/keypoints_semantic"
SEQ_DIR = "data/MERL/sequences_semantic"

for fn in sorted(os.listdir(KP_DIR)):
    if not fn.endswith(".npy"):
        continue
    path = os.path.join(KP_DIR, fn)
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        print(fn, "-> np.load error:", e); continue
    if arr is None or arr.size == 0 or (isinstance(arr.shape, tuple) and 0 in arr.shape):
        seq_name = fn.replace("_kps.npy","").replace(".npy","")
        seq_folder = os.path.join(SEQ_DIR, seq_name)
        if os.path.isdir(seq_folder):
            frames = [f for f in os.listdir(seq_folder) if f.endswith(".jpg")]
            print(fn, "BAD -> sequence folder exists:", seq_folder, "frame_count=", len(frames))
        else:
            print(fn, "BAD -> sequence folder missing:", seq_folder)
