# tools/list_bad_seq_names.py
import os, numpy as np

KP_DIR = "data/MERL/keypoints_semantic"
OUT = "data/MERL/bad_sequences.txt"
bad = []
for fn in sorted(os.listdir(KP_DIR)):
    if not fn.endswith(".npy"): continue
    path = os.path.join(KP_DIR, fn)
    try:
        arr = np.load(path, allow_pickle=False)
        if arr is None or arr.size == 0 or (isinstance(arr.shape, tuple) and 0 in arr.shape):
            seq = fn.replace("_kps.npy","").replace(".npy","")
            bad.append(seq)
    except Exception:
        seq = fn.replace("_kps.npy","").replace(".npy","")
        bad.append(seq)
with open(OUT,"w") as fh:
    fh.write("\n".join(bad))
print("Wrote", OUT, "with", len(bad), "bad seqs")
