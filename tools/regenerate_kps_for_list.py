# tools/regenerate_kps_for_list.py
import os, numpy as np, cv2
import mediapipe as mp
from tqdm import tqdm

SEQ_DIR = "data/MERL/sequences_semantic"
KP_DIR = "data/MERL/keypoints_semantic"
LIST_FILE = "data/MERL/bad_sequences.txt"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

with open(LIST_FILE, "r") as fh:
    seqs = [l.strip() for l in fh if l.strip()]

for seq in seqs:
    sdir = os.path.join(SEQ_DIR, seq)
    if not os.path.isdir(sdir):
        print("Missing sequence folder:", sdir); continue
    frames = sorted([f for f in os.listdir(sdir) if f.endswith(".jpg")])
    all_kps = []
    for f in tqdm(frames, desc=f"Regenerate {seq}"):
        img = cv2.imread(os.path.join(sdir, f))
        if img is None:
            all_kps.append(np.zeros((33,4), dtype=float)); continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            all_kps.append(np.zeros((33,4), dtype=float))
        else:
            kps = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
            all_kps.append(kps)
    arr = np.array(all_kps, dtype=np.float32)
    np.save(os.path.join(KP_DIR, f"{seq}_kps.npy"), arr)
    print("Saved", seq, "->", arr.shape)

pose.close()
