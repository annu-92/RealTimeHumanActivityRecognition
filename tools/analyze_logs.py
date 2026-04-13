# tools/analyze_logs.py
import pandas as pd
import os

LOG_PATH = "logs/realtime_events.csv"

if not os.path.exists(LOG_PATH):
    print(f"No log file found at {LOG_PATH}. Run realtime_predict.py first.")
    exit(0)

df = pd.read_csv(LOG_PATH)

print("Total events logged:", len(df))
print("\nActions count:")
print(df["display_label"].value_counts())

print("\nHelp requests (asking for help):")
print(df[df["help_requested"] == 1].groupby("display_label")["help_requested"].count())

# Optional: events per source (webcam vs video file)
print("\nEvents per source:")
print(df["source"].value_counts())
