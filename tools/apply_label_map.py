# tools/apply_label_map.py
import pandas as pd
import os

RAW_ANN = "data/MERL/annotations_perframe.csv"
MAP_FILE = "data/MERL/label_map.csv"
OUT_FILE = "data/MERL/annotations_semantic.csv"

# 1. Check mapping file exists
if not os.path.exists(MAP_FILE):
    raise FileNotFoundError(
        f"\nERROR: Mapping file '{MAP_FILE}' not found.\n"
        "Please create or copy label_map.csv first.\n"
    )

print("Loading raw annotations...")
ann = pd.read_csv(RAW_ANN)

print("Loading label map...")
label_map_df = pd.read_csv(MAP_FILE)

# convert raw_label → semantic_label dictionary
label_map = dict(zip(label_map_df["raw_label"], label_map_df["semantic_label"]))

# 2. Apply mapping
def map_func(raw):
    return label_map.get(raw, "unknown")

print("Applying mapping...")
ann["semantic_label"] = ann["label"].apply(map_func)

# 3. Save output
ann[["video", "frame", "semantic_label"]].to_csv(OUT_FILE, index=False)

print(f"\nWrote: {OUT_FILE}")
print("\nSemantic label distribution:")
print(ann["semantic_label"].value_counts().head(50))
