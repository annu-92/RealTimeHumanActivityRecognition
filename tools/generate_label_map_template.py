import pandas as pd

# Load label counts (first column is the label name)
counts = pd.read_csv("data/MERL/label_counts.csv", index_col=0)

# Extract the label index (raw tlabs labels)
raw_labels = counts.index.tolist()

# Take top 200 most common raw labels
top = raw_labels[:200]

out = "data/MERL/label_map_template.csv"

# Write template CSV
with open(out, "w", encoding="utf-8") as fh:
    fh.write("raw_label,semantic_label\n")
    for r in top:
        fh.write(f"{r},\n")

print("Wrote template:", out)
