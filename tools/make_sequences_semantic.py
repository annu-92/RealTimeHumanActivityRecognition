# tools/make_sequences_semantic.py
import os, pandas as pd, json, shutil
from collections import Counter

SEQ_DIR = "data/MERL/sequences_semantic"
FRAMES_DIR = "data/MERL/frames"
META = json.load(open("data/MERL/videos_meta.json"))
os.makedirs(SEQ_DIR, exist_ok=True)

df = pd.read_csv("data/MERL/annotations_semantic.csv")
window = 32   # frames per sequence
stride = 16
seq_idx = 0

for vid, g in df.groupby("video"):
    g_frames = sorted(g['frame'].unique())
    maxf = META.get(vid, {}).get('frames', max(g_frames))
    i = 1
    while i + window - 1 <= maxf:
        window_frames = list(range(i, i + window))
        labels = g[g['frame'].isin(window_frames)]['semantic_label'].tolist()
        if labels:
            lab = Counter(labels).most_common(1)[0][0]
            # only keep if at least half the window is a single label
            if lab != "unknown" and labels.count(lab) >= window * 0.5:
                seq_idx += 1
                seq_name = f"seq_{seq_idx:06d}"
                seq_folder = os.path.join(SEQ_DIR, seq_name)
                os.makedirs(seq_folder, exist_ok=True)
                src_folder = os.path.join(FRAMES_DIR, vid)
                for fnum in window_frames:
                    src = os.path.join(src_folder, f"frame_{fnum:06d}.jpg")
                    dst = os.path.join(seq_folder, f"frame_{fnum:06d}.jpg")
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                # write the label
                with open(os.path.join(seq_folder, "label.txt"), "w", encoding="utf-8") as fh:
                    fh.write(lab)
        i += stride

print("Created sequences (semantic):", seq_idx)
