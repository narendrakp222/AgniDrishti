import pandas as pd

from sklearn.ensemble import IsolationForest

# Read classified dataset
df = pd.read_csv(
    "data/processed/classified_events.csv"
)

# Features for anomaly detection
features = df[
    [
        "frp",
        "brightness",
        "risk_score",
        "georisk_score"
    ]
]

# Isolation Forest model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

# Train model
model.fit(features)

# Predict anomalies
df["anomaly"] = model.predict(features)

# Convert labels
df["anomaly_status"] = df["anomaly"].map(
    {
        1: "Normal",
        -1: "Anomaly"
    }
)

# Show results
print(
    df[
        [
            "frp",
            "brightness",
            "risk_score",
            "georisk_score",
            "anomaly_status"
        ]
    ].head(10)
)

# Save output
df.to_csv(
    "data/processed/anomaly_dataset.csv",
    index=False
)

print(
    "\nAnomaly detection completed!"
)