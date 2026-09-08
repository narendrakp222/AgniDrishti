# import pandas as pd

# df = pd.read_csv("data/processed/alert_dataset.csv")
# df1 = pd.read_csv("data/processed/anomaly_dataset.csv")
# df2 = pd.read_csv("data/processed/classified_events.csv")
# df3 = pd.read_csv("data/processed/escalation_dataset.csv")
# df4 = pd.read_csv("data/processed/georisk_dataset.csv")
# df5 = pd.read_csv("data/processed/georisk_hotspots.csv")
# df6 = pd.read_csv("data/processed/hotspots_with_risk.csv")
# df7 = pd.read_csv("data/processed/india_hotspots.csv")

# # print(df["alert_level"].value_counts())

# print(df.head())
# print(df1.head())
# print(df2.head())
# print(df3.head())
# print(df4.head())
# print(df5.head())
# print(df6.head())
# print(df7.head())

import pandas as pd

df = pd.read_csv(
    "data/processed/notification_dataset.csv"
)

print(df.head())