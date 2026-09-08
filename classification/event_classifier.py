import pandas as pd

df = pd.read_csv(
    "data/processed/georisk_dataset.csv"
)

def classify_event(row):

    if (
        row["factory_km"] < 10
        or row["industry_km"] < 10
    ):
        return "Industrial Flare"

    elif (
        row["brightness"] > 350
        and row["frp"] > 20
    ):
        return "Forest Fire"

    elif (
        row["brightness"] > 330
        and row["frp"] < 10
    ):
        return "Agricultural Burning"

    else:
        return "Other"


df["event_type"] = df.apply(
    classify_event,
    axis=1
)

print(
    df[
        [
            "frp",
            "brightness",
            "event_type"
        ]
    ].head()
)

df.to_csv(
    "data/processed/classified_events.csv",
    index=False
)

print(
    "Event classification completed!"
)