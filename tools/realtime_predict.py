# tools/realtime_predict.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import mediapipe as mp
import argparse
import os
import csv
from datetime import datetime

from train_lstm import LSTMPredictor  # uses the same model as training


# ---------- Config ----------
SEQ_LEN = 32                 # number of frames per sequence (same as training)
CONF_THRESH = 0.6            # minimum confidence to trust a prediction
HISTORY_LEN = 15             # how many past predictions to keep
STABLE_REQUIRED = 7          # label must appear this many times in history to update display

MODEL_PATH = "models/final_lstm.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- CLI args ----------
parser = argparse.ArgumentParser(description="Real-time activity recognition (webcam or video)")
parser.add_argument("--video", type=str, default=None, help="Path to input video file. If not set, uses webcam.")
parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained model (.pt)")
parser.add_argument("--no-display", action="store_true", help="Disable imshow display (useful when running headless).")
parser.add_argument("--out", type=str, default=None, help="Path to write annotated output video (when --video is used).")
parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (useful for testing webcam).")
args = parser.parse_args()

# override model path if provided
MODEL_PATH = args.model

# ---------- Load model + label mapping ----------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

label2id = ckpt.get("label2id")
if label2id is None:
    raise RuntimeError(f"{MODEL_PATH} does not contain label2id mapping.")

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
help_message = ""       # transient help message when hand raised

# ---------- Logging setup ----------
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "realtime_events.csv")

# open in append mode
log_file = open(log_path, "a", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)

# write header only if file is empty
if os.stat(log_path).st_size == 0:
    log_writer.writerow([
        "timestamp",
        "source",
        "frame_idx",
        "raw_label",
        "display_label",
        "probability",
        "help_requested"
    ])


# ---------- Video loop ----------
if args.video:
    cap = cv2.VideoCapture(args.video)
    source_desc = args.video
    # prepare output writer
    out_path = args.out if args.out else None
    if out_path is None:
        # default output filename next to input
        base = args.video
        if "." in base:
            out_path = base.rsplit('.', 1)[0] + "_annotated.mp4"
        else:
            out_path = base + "_annotated.mp4"
    writer = None
else:
    cap = cv2.VideoCapture(0)
    source_desc = "webcam"
    writer = None

if not cap.isOpened():
    log_file.close()
    raise RuntimeError(f"Could not open video source: {source_desc}")

print(f"Real-time activity recognition running on {source_desc}. Press 'q' to quit.")

frame_idx = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_idx += 1

    # stop early if requested (useful for testing webcam)
    if args.max_frames is not None and frame_idx >= args.max_frames:
        break

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose estimation
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # ---------- Raised-hand detection ----------
        # MediaPipe Pose landmark indices: 11=LEFT_SHOULDER, 12=RIGHT_SHOULDER,
        # 13=LEFT_ELBOW, 14=RIGHT_ELBOW, 15=LEFT_WRIST, 16=RIGHT_WRIST.
        try:
            l_sh = landmarks[11]
            r_sh = landmarks[12]
            l_el = landmarks[13]
            r_el = landmarks[14]
            l_wr = landmarks[15]
            r_wr = landmarks[16]

            h_px = frame.shape[0]
            vis_thresh = 0.2
            margin_px = int(0.04 * h_px)  # how many pixels wrist/elbow must be above shoulder

            def is_raised(sh, el, wr):
                sh_y = int(sh.y * h_px)
                el_y = int(el.y * h_px)
                wr_y = int(wr.y * h_px)
                # check wrist first (preferred)
                if wr.visibility > vis_thresh and sh.visibility > vis_thresh:
                    if wr_y < sh_y - margin_px:
                        return True
                # fallback to elbow if wrist not visible or occluded
                if el.visibility > vis_thresh and sh.visibility > vis_thresh:
                    if el_y < sh_y - margin_px:
                        return True
                return False

            left_raised = is_raised(l_sh, l_el, l_wr)
            right_raised = is_raised(r_sh, r_el, r_wr)
            if left_raised or right_raised:
                help_message = "Asking for help"
            else:
                help_message = ""
        except Exception:
            help_message = ""

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
                raw_label = id2label[pred_idx]
                PRED_HISTORY.append(raw_label)

                # If we have enough history, update display_label using majority vote
                if len(PRED_HISTORY) >= STABLE_REQUIRED:
                    labels_arr, counts = np.unique(list(PRED_HISTORY), return_counts=True)
                    majority_label = labels_arr[counts.argmax()]
                    majority_count = counts.max()

                    if majority_count >= STABLE_REQUIRED:
                        display_label = majority_label

                # --------- Log this prediction ---------
                timestamp = datetime.now().isoformat(timespec="seconds")
                log_writer.writerow([
                    timestamp,
                    source_desc,
                    frame_idx,
                    raw_label,
                    display_label,
                    f"{max_prob:.4f}",
                    1 if help_message else 0,
                ])
            else:
                # Low confidence: you can optionally log as uncertain or skip
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

    # Draw help message if hand raised
    if help_message:
        # draw background box
        (w_text, h_text), _ = cv2.getTextSize(help_message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        x = 20
        y = 80
        cv2.rectangle(frame, (x-8, y-28), (x + w_text + 8, y + 8), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, help_message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # write annotated frame if writer available (video input)
    if args.video:
        if writer is None:
            # initialize writer with same size/fps as input
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(frame)

    if not args.no_display:
        cv2.imshow("Real-Time Activity Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # when display disabled, still allow loop to run until video ends
        pass

cap.release()
if 'writer' in locals() and writer is not None:
    writer.release()
cv2.destroyAllWindows()
pose.close()
log_file.close()
print(f"Events logged to {log_path}")
