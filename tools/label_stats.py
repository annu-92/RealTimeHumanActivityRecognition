import pandas as pd

df = pd.read_csv("data/MERL/annotations_perframe.csv")
counts = df['label'].value_counts()

print("Unique raw labels:", len(counts))
print(counts.head(40))

counts.to_csv("data/MERL/label_counts.csv", header=["count"])
print("Wrote data/MERL/label_counts.csv")
