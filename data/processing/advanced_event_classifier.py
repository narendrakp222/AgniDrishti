from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "georisk_dataset.csv"
OUTPUT_PATH = DATA_DIR / "classified_events.csv"


def load_georisk_data():
    if INPUT_PATH.exists():
        return pd.read_csv(INPUT_PATH)
    return pd.read_csv(DATA_DIR / "escalation_dataset.csv")


def classify(row):
    if (
        row["factory_km"] < 20
        and row["brightness"] > 350
        and row["frp"] > 10
    ):
        return "Industrial Fire"
    elif row["powerplant_km"] < 20:
        return "Thermal Power Plant"
    elif (
        row["industry_km"] < 20
        and row["brightness"] > 340
    ):
        return "Gas Flare"
    elif (
        row["brightness"] >= 350
        and row["frp"] >= 8
    ):
        return "Forest Fire"
    elif (
        row["brightness"] >= 348
        and row["frp"] >= 5
    ):
        return "Gas Flare"
    elif row["frp"] >= 8:
        return "Mining Activity"
    elif (
        row["brightness"] < 340
        and row["frp"] < 6
    ):
        return "Agricultural Burning"
    return "Unknown Thermal Source"


def create_classifications():
    df = load_georisk_data()
    df["event_type"] = df.apply(classify, axis=1)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    classified = create_classifications()
    print(classified["event_type"].value_counts())
