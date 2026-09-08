from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "escalation_dataset.csv"
OUTPUT_PATH = DATA_DIR / "population_dataset.csv"


def load_escalation_data():
    if INPUT_PATH.exists():
        return pd.read_csv(INPUT_PATH)

    df = pd.read_csv(DATA_DIR / "india_hotspots.csv")
    frp_score = df["frp"].clip(upper=80) / 80
    brightness_score = (df["brightness"] - 295) / (367 - 295)
    df["risk_score"] = ((0.6 * frp_score + 0.4 * brightness_score) * 100).round(2)
    df["georisk_score"] = df["risk_score"]
    df["escalation_level"] = pd.cut(
        df["georisk_score"],
        bins=[-float("inf"), 40, 70, 100, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ).astype(str)
    return df


def exposure_level(population):
    if population > 30000:
        return "HIGH"
    elif population > 10000:
        return "MEDIUM"
    return "LOW"


def create_population_dataset():
    df = load_escalation_data()
    np.random.seed(42)
    df["population_exposed"] = np.random.randint(1000, 50000, len(df))
    df["exposure_level"] = df["population_exposed"].apply(exposure_level)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    create_population_dataset()
    print("Population exposure dataset created!")
