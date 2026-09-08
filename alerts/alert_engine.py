import pandas as pd

df = pd.read_csv(
    "data/processed/georisk_dataset.csv"
)

def get_alert(score):

    if score >= 120:
        return "CRITICAL"

    elif score >= 80:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    else:
        return "LOW"

df["alert_level"] = df["georisk_score"].apply(
    get_alert
)

messages = {
    "CRITICAL": "Immediate response required",
    "HIGH": "High wildfire risk detected",
    "MEDIUM": "Monitor hotspot closely",
    "LOW": "No immediate threat"
}

df["alert_message"] = df["alert_level"].map(
    messages
)

print(
    df[
        [
            "latitude",
            "longitude",
            "georisk_score",
            "alert_level"
        ]
    ].head()
)

df.to_csv(
    "data/processed/alert_dataset.csv",
    index=False
)

print("Alert dataset saved successfully!")