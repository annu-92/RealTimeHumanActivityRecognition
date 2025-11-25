# tools/fix_annotation_video_ids.py
import pandas as pd
import os, shutil

IN = "data/MERL/annotations_semantic.csv"
OUT = "data/MERL/annotations_semantic_fixed.csv"
BACKUP = "data/MERL/annotations_semantic.backup.csv"

if not os.path.exists(IN):
    raise SystemExit(f"Input file not found: {IN}")

# backup original
shutil.copyfile(IN, BACKUP)
print("Backup written to", BACKUP)

df = pd.read_csv(IN)
orig_videos = sorted(df['video'].unique())[:50]
print("Sample original video ids:", orig_videos[:10])

# rule: if video ends with '_label', remove that suffix
def fix_vid(v):
    if isinstance(v, str) and v.endswith("_label"):
        return v[:-6]  # drop "_label"
    return v

df['video'] = df['video'].apply(fix_vid)

fixed_videos = sorted(df['video'].unique())[:50]
print("Sample fixed video ids:", fixed_videos[:10])

# count how many rows changed
orig = pd.read_csv(BACKUP)
changed = (orig['video'] != df['video']).sum()
print(f"Rows changed: {changed} / {len(df)}")

# save
df.to_csv(OUT, index=False)
print("Wrote fixed annotations to", OUT)

# optional: overwrite original (commented out — do this only if you're happy)
# shutil.copyfile(OUT, IN)
# print("Overwrote original annotations_semantic.csv with fixed versions.")
