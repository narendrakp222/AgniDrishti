from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escalation_predictor import load_escalation_data


df = load_escalation_data()

label_map = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3
}

df["target"] = df["escalation_level"].map(label_map)

features = [
    "frp",
    "brightness",
    "risk_score",
    "georisk_score"
]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Complete")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
