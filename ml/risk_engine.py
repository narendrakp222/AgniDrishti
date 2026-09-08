import pandas as pd

# Read cleaned dataset
df = pd.read_csv("data/processed/clean_india_hotspots.csv")


# Risk calculation function
def calculate_risk(frp, brightness):

    frp_score = min(frp, 80) / 80

    brightness_score = (brightness - 295) / (367 - 295)

    risk = (0.6 * frp_score + 0.4 * brightness_score) * 100

    return round(risk, 2)


# Apply risk score
df["risk_score"] = df.apply(
    lambda x: calculate_risk(
        x["frp"],
        x["brightness"]
    ),
    axis=1
)

# Risk level
def risk_level(score):

    if score >= 70:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


df["risk_level"] = df["risk_score"].apply(risk_level)


# Save dataset
df.to_csv(
    "data/processed/hotspots_with_risk.csv",
    index=False
)

print(df[
    ["latitude", "longitude", "risk_score", "risk_level"]
].head())

print("\nRisk Distribution:")
print(df["risk_level"].value_counts())

high_risk = df[df["risk_level"] == "High"]

print("\nHigh Risk Hotspots:")
print(
    high_risk[
        ["latitude", "longitude", "frp", "brightness", "risk_score"]
    ].head(10)
)

print("\nRisk dataset saved successfully!")