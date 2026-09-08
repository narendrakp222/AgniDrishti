from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "escalation_dataset.csv"
OUTPUT_PATH = DATA_DIR / "prediction_dataset.csv"


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


def predict_probability(row):
    score = row["georisk_score"]

    if score >= 100:
        return 95
    elif score >= 90:
        return 85
    elif score >= 75:
        return 70
    elif score >= 60:
        return 50
    return 25


def generate_predictions():
    df = load_escalation_data()
    df["escalation_probability"] = df.apply(
        predict_probability,
        axis=1
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    predictions = generate_predictions()
    print(
        predictions[
            [
                "georisk_score",
                "escalation_level",
                "escalation_probability"
            ]
        ].head()
    )
