# tools/extract_keypoints_mediapipe_sequences.py
import os, numpy as np, cv2
import mediapipe as mp
from tqdm import tqdm

SEQ_DIR = "data/MERL/sequences_semantic"
KP_DIR = "data/MERL/keypoints_semantic"
os.makedirs(KP_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

for seq in sorted(os.listdir(SEQ_DIR)):
    sdir = os.path.join(SEQ_DIR, seq)
    if not os.path.isdir(sdir):
        continue
    frames = sorted([f for f in os.listdir(sdir) if f.endswith(".jpg")])
    all_kps = []
    for f in tqdm(frames, desc=f"KP {seq}"):
        img = cv2.imread(os.path.join(sdir, f))
        if img is None:
            all_kps.append(np.zeros((33,4)))
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            all_kps.append(np.zeros((33,4)))
        else:
            kps = []
            for lm in res.pose_landmarks.landmark:
                kps.append([lm.x, lm.y, lm.z, lm.visibility])
            all_kps.append(kps)
    arr = np.array(all_kps, dtype=np.float32)  # (T, 33, 4)
    np.save(os.path.join(KP_DIR, f"{seq}_kps.npy"), arr)

pose.close()
print("Keypoints extraction done.")
