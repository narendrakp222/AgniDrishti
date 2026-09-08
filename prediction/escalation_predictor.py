import pandas as pd

df = pd.read_csv(
    "data/processed/anomaly_dataset.csv"
)

def predict_escalation(row):

    if (
        row["georisk_score"] > 120
        or row["anomaly_status"] == "Anomaly"
    ):
        return "CRITICAL"

    elif row["georisk_score"] > 90:
        return "HIGH"

    elif row["georisk_score"] > 60:
        return "MEDIUM"

    else:
        return "LOW"


df["escalation_level"] = df.apply(
    predict_escalation,
    axis=1
)

print(
    df[
        [
            "georisk_score",
            "anomaly_status",
            "escalation_level"
        ]
    ].head(10)
)

df.to_csv(
    "data/processed/escalation_dataset.csv",
    index=False
)

print(
    "\nEscalation prediction completed!"
)