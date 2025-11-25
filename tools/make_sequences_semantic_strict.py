# tools/make_sequences_semantic_strict.py
import os, pandas as pd, json, shutil
from collections import Counter

SEQ_DIR = "data/MERL/sequences_semantic"
FRAMES_DIR = "data/MERL/frames"
META_FILE = "data/MERL/videos_meta.json"
LOGFILE = "data/MERL/make_sequences_log.txt"

meta = json.load(open(META_FILE)) if os.path.exists(META_FILE) else {}
os.makedirs(SEQ_DIR, exist_ok=True)

df = pd.read_csv("data/MERL/annotations_semantic.csv")
window = 32
stride = 16
seq_idx = 0
skipped = []

with open(LOGFILE, "w", encoding="utf-8") as log:
    for vid, g in df.groupby("video"):
        g_frames = sorted(g['frame'].unique())
        maxf = int(meta.get(vid, {}).get('frames', max(g_frames)))
        i = 1
        while i + window - 1 <= maxf:
            window_frames = list(range(i, i + window))
            labels = g[g['frame'].isin(window_frames)]['semantic_label'].tolist()
            if labels:
                lab = Counter(labels).most_common(1)[0][0]
                if lab != "unknown" and labels.count(lab) >= window * 0.5:
                    # check all frames exist
                    src_folder = os.path.join(FRAMES_DIR, vid)
                    missing = []
                    for fnum in window_frames:
                        src = os.path.join(src_folder, f"frame_{fnum:06d}.jpg")
                        if not os.path.exists(src):
                            missing.append(src)

                    if missing:
                        skipped.append((vid, window_frames[0], window_frames[-1], len(missing)))
                        log.write(f"SKIP {vid} {i}-{i+window-1} missing {len(missing)} frames\n")
                    else:
                        seq_idx += 1
                        seq_name = f"seq_{seq_idx:06d}"
                        seq_folder = os.path.join(SEQ_DIR, seq_name)
                        os.makedirs(seq_folder, exist_ok=True)

                        for fnum in window_frames:
                            src = os.path.join(src_folder, f"frame_{fnum:06d}.jpg")
                            dst = os.path.join(seq_folder, f"frame_{fnum:06d}.jpg")
                            shutil.copy(src, dst)

                        with open(os.path.join(seq_folder, "label.txt"), "w", encoding="utf-8") as fh:
                            fh.write(lab)

            i += stride

    log.write(f"\nCreated sequences: {seq_idx}\n")
    log.write(f"Skipped windows (missing frames): {len(skipped)}\n")

print("Done. Created sequences:", seq_idx)
print("Skipped windows:", len(skipped))
print("Log saved to:", LOGFILE)
