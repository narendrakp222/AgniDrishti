from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "weather_dataset.csv"
OUTPUT_PATH = DATA_DIR / "infrastructure_threat_dataset.csv"


def threat_level(distance):
    if distance < 20:
        return "CRITICAL"
    elif distance < 50:
        return "HIGH"
    elif distance < 100:
        return "MEDIUM"
    return "LOW"


def create_threat_dataset():
    df = pd.read_csv(INPUT_PATH)
    df["hospital_threat"] = df["hospital_km"].apply(threat_level)
    df["industry_threat"] = df["industry_km"].apply(threat_level)
    df["factory_threat"] = df["factory_km"].apply(threat_level)
    df["powerplant_threat"] = df["powerplant_km"].apply(threat_level)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    create_threat_dataset()
    print("Infrastructure threat dataset created!")
