# tools/realtime_predict.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import mediapipe as mp

from train_lstm import LSTMPredictor  # uses the same model as training


# ---------- Config ----------
SEQ_LEN = 32                 # number of frames per sequence (same as training)
CONF_THRESH = 0.6            # minimum confidence to trust a prediction
HISTORY_LEN = 15             # how many past predictions to keep
STABLE_REQUIRED = 7          # label must appear this many times in history to update display

MODEL_PATH = "models/final_lstm.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Load model + label mapping ----------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

label2id = ckpt.get("label2id")
if label2id is None:
    raise RuntimeError("final_lstm.pt does not contain label2id mapping.")

# id2label: int -> string label
id2label = {v: k for k, v in label2id.items()}
num_classes = len(label2id)

# Input size: we trained with x,y only -> 33 landmarks * 2 = 66
INPUT_SIZE = 66

model = LSTMPredictor(
    input_size=INPUT_SIZE,
    hidden_size=256,
    num_layers=2,
    num_classes=num_classes,
    bidirectional=True,
    dropout=0.4,
)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()


# ---------- MediaPipe setup ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ---------- Helper state ----------
# stores last SEQ_LEN frames of keypoints (for a single sliding window)
WINDOW = deque(maxlen=SEQ_LEN)

# stores last HISTORY_LEN predicted labels for smoothing
PRED_HISTORY = deque(maxlen=HISTORY_LEN)

display_label = "..."   # what we actually draw on screen


# ---------- Video loop ----------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Real-time activity recognition running. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose estimation
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Take x,y only (same as training), flatten to 66-dim vector
        kp_xy = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32).flatten()
        WINDOW.append(kp_xy)

        # Only run model once we have a full sequence
        if len(WINDOW) == SEQ_LEN:
            seq_arr = np.stack(WINDOW, axis=0)  # shape: (T, 66)

            # Per-sequence normalization (same as training)
            mean = seq_arr.mean(axis=0, keepdims=True)
            std = seq_arr.std(axis=0, keepdims=True) + 1e-6
            seq_arr = (seq_arr - mean) / std

            x = torch.from_numpy(seq_arr).unsqueeze(0).to(DEVICE)  # (1, T, F)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                max_prob, pred_idx = probs.max(dim=1)
                max_prob = max_prob.item()
                pred_idx = pred_idx.item()

            # Only trust prediction if confidence is high enough
            if max_prob >= CONF_THRESH:
                pred_label = id2label[pred_idx]
                PRED_HISTORY.append(pred_label)

                # If we have enough history, update display_label using majority vote
                if len(PRED_HISTORY) >= STABLE_REQUIRED:
                    labels, counts = np.unique(list(PRED_HISTORY), return_counts=True)
                    majority_label = labels[counts.argmax()]
                    majority_count = counts.max()

                    if majority_count >= STABLE_REQUIRED:
                        display_label = majority_label
            else:
                # Low confidence: you can optionally show "idle" or "uncertain"
                # display_label = "uncertain"
                pass

    # Draw the *smoothed* label on the frame
    cv2.putText(
        frame,
        f"Action: {display_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Real-Time Activity Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
