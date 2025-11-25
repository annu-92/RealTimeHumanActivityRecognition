# tools/find_bad_kps.py
import os, numpy as np

KP_DIR = "data/MERL/keypoints_semantic"
bad = []
for fn in sorted(os.listdir(KP_DIR)):
    if not fn.endswith(".npy"):
        continue
    path = os.path.join(KP_DIR, fn)
    try:
        arr = np.load(path, allow_pickle=False)
        if arr is None or arr.size == 0 or (isinstance(arr.shape, tuple) and 0 in arr.shape):
            bad.append((fn, arr.shape if hasattr(arr,'shape') else None))
    except Exception as e:
        bad.append((fn, f"error:{e}"))
if bad:
    print("Found bad keypoint files:", len(bad))
    for b in bad:
        print(b[0], "->", b[1])
else:
    print("No bad keypoint files found.")
