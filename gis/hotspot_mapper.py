import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path

DATA_DIR = Path("data/processed")
df = pd.read_csv(DATA_DIR / "escalation_dataset.csv")
classified_path = DATA_DIR / "classified_events.csv"
if classified_path.exists():
    classified = pd.read_csv(classified_path)
    df["event_type"] = classified["event_type"]

weather_path = DATA_DIR / "weather_dataset.csv"
if weather_path.exists():
    weather = pd.read_csv(weather_path)
    df["weather_risk_score"] = weather["weather_risk_score"]
    df["population_exposed"] = weather.get("population_exposed", 0)
else:
    df["weather_risk_score"] = 0
    df["population_exposed"] = 0


def weather_level(score):
    if score >= 25:
        return "HIGH"
    if score >= 15:
        return "MEDIUM"
    return "LOW"

m = folium.Map(
    location=[22, 80],
    zoom_start=5
)

for _, row in df.iterrows():

    if row["escalation_level"] == "CRITICAL":
        color = "red"

    elif row["escalation_level"] == "HIGH":
        color = "orange"

    elif row["escalation_level"] == "MEDIUM":
        color = "yellow"

    else:
        color = "green"

    anomaly = "Yes" if row["anomaly_status"] == "Anomaly" else "No"
    popup_text = f"""
    <b>Event:</b> {row['event_type']}<br>
    <b>GeoRisk:</b> {row['georisk_score']}<br>
    <b>Escalation:</b> {row['escalation_level']}<br>
    <b>Anomaly:</b> {anomaly}<br>
    <b>Weather Risk:</b> {weather_level(row['weather_risk_score'])}<br>
    <b>Population Exposed:</b> {int(row['population_exposed'])}<br>
    <hr>
    <b>Nearest Hospital:</b> {row['hospital_km']} km<br>
    <b>Nearest Industry:</b> {row['industry_km']} km<br>
    <b>Nearest Factory:</b> {row['factory_km']} km<br>
    <b>Nearest Power Plant:</b> {row['powerplant_km']} km
    """

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        popup=popup_text
    ).add_to(m)

heat_data = [
    [row["latitude"], row["longitude"], row["georisk_score"]]
    for _, row in df.iterrows()
]

HeatMap(
    heat_data,
    name="Risk Heatmap"
).add_to(m)

folium.LayerControl().add_to(m)

legend_html = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 220px;
height: 130px;
background-color: white;
border:2px solid black;
z-index:9999;
padding:10px;
">

<h4>Escalation Levels</h4>

🔴 Critical<br>
🟠 High<br>
🟡 Medium<br>
🟢 Low

</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)

m.save(
    "templates/hotspot_map.html"
)

print("Map generated!")