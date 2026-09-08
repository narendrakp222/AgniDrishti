from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "probability_dataset.csv"

sys.path.insert(0, str(ROOT_DIR))
from src.prediction.escalation_predictor import load_escalation_data


df = load_escalation_data()


def calculate_probability(row):
    score = row["georisk_score"]

    if row.get("anomaly_status") == "Anomaly":
        score += 10

    return round(min(score, 100), 2)


df["escalation_probability"] = df.apply(
    calculate_probability,
    axis=1
)

df.to_csv(OUTPUT_PATH, index=False)

print("Probability dataset created!")
