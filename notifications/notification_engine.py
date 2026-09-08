import pandas as pd

df = pd.read_csv(
    "data/processed/escalation_dataset.csv"
)

notifications = []

for _, row in df.iterrows():

    if row["escalation_level"] == "CRITICAL":

        recipient = "National Disaster Authority"

        message = (
            f"CRITICAL ALERT: "
            f"{row['event_type']} detected "
            f"at ({row['latitude']}, {row['longitude']}) "
            f"GeoRisk={row['georisk_score']}"
        )

    elif row["escalation_level"] == "HIGH":

        recipient = "State Emergency Center"

        message = (
            f"HIGH ALERT: "
            f"{row['event_type']} hotspot detected."
        )

    elif row["escalation_level"] == "MEDIUM":

        recipient = "District Control Room"

        message = (
            f"MEDIUM ALERT: "
            f"Monitor hotspot activity."
        )

    else:

        recipient = "Local Authority"

        message = (
            f"LOW ALERT: Routine monitoring."
        )

    notifications.append({

        "latitude":
        row["latitude"],

        "longitude":
        row["longitude"],

        "event_type":
        row["event_type"],

        "georisk_score":
        row["georisk_score"],

        "escalation_level":
        row["escalation_level"],

        "recipient":
        recipient,

        "notification_message":
        message,

        "status":
        "PENDING"
    })

notification_df = pd.DataFrame(
    notifications
)

notification_df.to_csv(
    "data/processed/notification_dataset.csv",
    index=False
)

print(
    notification_df.head()
)

print(
    "\nNotification dataset created!"
)