# tools/convert_tlabs_mat.py
import os
import json
import csv
import numpy as np
import scipy.io

MAT_DIR = "data/MERL/annotations"
META_FILE = "data/MERL/videos_meta.json"   # use fps info if available
OUT_CSV = "data/MERL/annotations_perframe.csv"
INSPECT_OUTPUT = "data/MERL/tlabs_inspect_sample.txt"

# load video meta (fps) if available
if os.path.exists(META_FILE):
    with open(META_FILE, "r") as fh:
        video_meta = json.load(fh)
else:
    video_meta = {}

def is_seconds_like(arr):
    """Heuristic: if max value < 20 and has floats, likely seconds.
       If values are large (>100) or integer-like, treat as frames."""
    if arr.size == 0:
        return False
    a = np.asarray(arr).ravel()
    # ignore NaNs
    a = a[~np.isnan(a)]
    if a.size == 0:
        return False
    # if any value > 1000 -> definitely frames
    if np.any(a > 1000):
        return False
    # if max <= 20 and any fractional -> seconds
    if a.max() <= 20 and (np.any(np.mod(a,1) != 0) or a.dtype.kind == 'f'):
        return True
    # otherwise assume frames
    return False

def inspect_and_convert():
    mats = sorted([f for f in os.listdir(MAT_DIR) if f.lower().endswith(".mat")])
    if not mats:
        print("No .mat files found in", MAT_DIR)
        return

    out_rows = []
    inspect_lines = []
    for matfile in mats:
        path = os.path.join(MAT_DIR, matfile)
        base = os.path.splitext(matfile)[0]
        try:
            data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print("Failed to load", matfile, ":", e)
            continue
        # expected top-level key 'tlabs'
        if 'tlabs' not in data:
            print(f"Warning: 'tlabs' not found in {matfile}, keys:", [k for k in data.keys() if not k.startswith("__")])
            continue
        tlabs = data['tlabs']  # this is an ndarray of objects (as you inspected)
        inspect_lines.append("="*80)
        inspect_lines.append(f"FILE: {matfile}  tlabs type={type(tlabs)}, shape={getattr(tlabs,'shape',None)}")
        # iterate groups
        # tlabs is shape (5,), dtype=object — loop through its elements
        for gi, group in enumerate(tlabs):
            # group is typically an ndarray shape (N,2)
            arr = np.array(group)
            inspect_lines.append(f" Group {gi}: arr.shape={arr.shape}, dtype={arr.dtype}")
            # show first 6 rows as sample
            sample_rows = arr[:6].tolist() if arr.size>0 else []
            inspect_lines.append(f"  sample rows: {sample_rows}")
            # detect seconds vs frames
            use_seconds = is_seconds_like(arr)
            inspect_lines.append(f"  heuristic -> use_seconds={use_seconds}")
            # convert each row to frame ranges
            for si, row in enumerate(arr):
                # guard shape: row may be 1D length 2 or length>2
                if hasattr(row, "__len__") and len(row) >= 2:
                    s = float(row[0])
                    e = float(row[1])
                else:
                    # unexpected row shape: skip
                    continue
                fps = video_meta.get(base, {}).get("fps", 30)
                if use_seconds:
                    start_f = int(round(s * fps))
                    end_f   = int(round(e * fps))
                else:
                    start_f = int(round(s))
                    end_f   = int(round(e))
                if start_f < 1:
                    start_f = 1
                if end_f < start_f:
                    end_f = start_f
                label = f"tlabs_g{gi}_s{si}"
                # append per-frame rows
                for fr in range(start_f, end_f + 1):
                    out_rows.append([base, fr, label])
        inspect_lines.append("")  # blank line between files

    # write inspect sample for your review
    with open(INSPECT_OUTPUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(inspect_lines))
    print("Wrote inspection sample to", INSPECT_OUTPUT)

    # write CSV
    if out_rows:
        with open(OUT_CSV, "w", newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(["video","frame","label"])
            writer.writerows(out_rows)
        print("Wrote per-frame annotations to", OUT_CSV, " (rows:", len(out_rows), ")")
    else:
        print("No rows produced; check tlabs content.")

if __name__ == "__main__":
    inspect_and_convert()
