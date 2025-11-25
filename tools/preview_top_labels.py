# tools/preview_top_labels.py
import os
import cv2
import pandas as pd

# number of labels to preview (change this if you want more/less)
TOP_N = 40

# playback speed (delay between frames)
PLAYBACK_FPS = 15   # higher = faster preview

# read label counts (no squeeze needed)
counts = pd.read_csv("data/MERL/label_counts.csv", index_col=0)

# list of raw labels to preview
labels = list(counts.index[:TOP_N])

# read full annotations
ann = pd.read_csv("data/MERL/annotations_perframe.csv")

def get_first_segment(raw_label):
    """Return (video, frame_list) for first non-empty occurrence of raw_label."""
    subset = ann[ann["label"] == raw_label]
    if subset.empty:
        return None, None
    for vid, g in subset.groupby("video"):
        frames = sorted(g["frame"].unique())
        return vid, frames
    return None, None

for raw in labels:
    vid, frames = get_first_segment(raw)
    if vid is None or not frames:
        print(f"No frames found for {raw}")
        continue
    
    print(f"\n=== Previewing {raw} in video {vid} ===")
    print(f"Frame range: {frames[0]} – {frames[-1]}")
    
    frame_dir = os.path.join("data", "MERL", "frames", vid)
    
    # show at most ~60 frames
    preview_frames = frames[:60]

    for fr in preview_frames:
        img_path = os.path.join(frame_dir, f"frame_{fr:06d}.jpg")
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        text = f"{raw} | {vid} | frame {fr}"
        cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 2)
        
        cv2.imshow("Preview", img)
        
        # waitKey timing controls playback speed  
        if cv2.waitKey(int(1000/PLAYBACK_FPS)) & 0xFF == ord('q'):
            break
    
    print("Press ANY key for next label, or 'q' in the image window to stop...")
    k = cv2.waitKey(0)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("\nFinished previewing labels.")
