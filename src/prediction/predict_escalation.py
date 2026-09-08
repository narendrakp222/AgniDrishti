from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escalation_predictor import load_escalation_data


df = load_escalation_data()


def probability(row):
    score = row["georisk_score"]
    return min(99, round(score))


df["escalation_probability"] = df.apply(
    probability,
    axis=1
)

df.to_csv(
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "prediction_dataset.csv",
    index=False
)

print("Prediction dataset created!")
