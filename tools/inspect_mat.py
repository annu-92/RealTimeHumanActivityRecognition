import os
import sys
import scipy.io
import numpy as np

MAT_DIR = "data/MERL/annotations"

def safe_repr(obj):
    try:
        if isinstance(obj, np.ndarray):
            return f"ndarray shape={obj.shape}, dtype={obj.dtype}"
        return repr(obj)
    except:
        return str(type(obj))

def inspect_mat(path):
    print("=" * 80)
    print("FILE:", path)

    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in data.keys() if not k.startswith("__")]

    print("Top-level keys:", keys)

    for k in keys:
        v = data[k]
        print(f"\n--- Key: {k} | Type: {type(v)}")
        print("Repr:", safe_repr(v))

        # If value has __dict__ (MATLAB struct)
        if hasattr(v, "__dict__"):
            print("Fields:", list(vars(v).keys()))

        # If numpy structured array
        if isinstance(v, np.ndarray) and v.dtype.names:
            print("Structured array fields:", v.dtype.names)

        # Show one example from object array
        if isinstance(v, np.ndarray) and v.size > 0:
            example = v.flat[0]
            print("Example element type:", type(example))
            try:
                if hasattr(example, "__dict__"):
                    print("Example fields:", list(vars(example).keys()))
                elif isinstance(example, np.ndarray):
                    print("Example ndarray shape:", example.shape)
                else:
                    print("Example repr:", safe_repr(example))
            except:
                pass


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(MAT_DIR) if f.lower().endswith(".mat"))
    if not files:
        print("No .mat files found in", MAT_DIR)
        sys.exit()

    for f in files[:3]:
        inspect_mat(os.path.join(MAT_DIR, f))

    print("\nFinished inspecting .mat files.")
