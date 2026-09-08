from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "temporal_analysis.csv"


def load_escalation_data():
    escalation_path = DATA_DIR / "escalation_dataset.csv"
    if escalation_path.exists():
        return pd.read_csv(escalation_path)

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


def create_temporal_analysis():
    df = load_escalation_data()
    df["date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")

    summary = (
        df.groupby("date", as_index=False)
        .agg(
            total_hotspots=("georisk_score", "size"),
            critical_count=("escalation_level", lambda values: (values == "CRITICAL").sum()),
            high_count=("escalation_level", lambda values: (values == "HIGH").sum()),
            medium_count=("escalation_level", lambda values: (values == "MEDIUM").sum()),
            avg_georisk=("georisk_score", "mean")
        )
        .sort_values("date")
    )
    summary["avg_georisk"] = summary["avg_georisk"].round(2)
    summary.to_csv(OUTPUT_PATH, index=False)
    return summary


if __name__ == "__main__":
    create_temporal_analysis()
    print("Temporal dataset created!")
