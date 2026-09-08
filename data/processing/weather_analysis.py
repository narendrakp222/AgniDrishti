from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "population_dataset.csv"
OUTPUT_PATH = DATA_DIR / "weather_dataset.csv"


def create_weather_dataset():
    df = pd.read_csv(INPUT_PATH)

    np.random.seed(42)
    df["temperature"] = np.random.randint(25, 45, len(df))
    df["humidity"] = np.random.randint(20, 90, len(df))
    df["wind_speed"] = np.random.randint(5, 40, len(df))
    df["rainfall"] = np.random.randint(0, 50, len(df))

    weather_score = []
    for _, row in df.iterrows():
        score = 0
        if row["temperature"] > 35:
            score += 10
        if row["humidity"] < 30:
            score += 10
        if row["wind_speed"] > 20:
            score += 15
        if row["rainfall"] > 10:
            score -= 10
        weather_score.append(score)

    df["weather_risk_score"] = weather_score
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    create_weather_dataset()
    print("Weather dataset created!")
