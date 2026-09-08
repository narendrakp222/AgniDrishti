from geopy.distance import geodesic
import pandas as pd
import geopandas as gpd

# Load hotspots
hotspots = pd.read_csv(
    "data/processed/hotspots_with_risk.csv"
)

# Load hospitals
hospitals = gpd.read_file(
    "data/raw/hospitals.geojson"
)

# Test first hotspot
hotspot = hotspots.iloc[0]

min_distance = float("inf")
nearest_hospital = "Unknown"
for _, hospital in hospitals.iterrows():

    hospital_point = hospital.geometry.centroid

    hospital_lat = hospital_point.y
    hospital_lon = hospital_point.x

    distance = geodesic(
        (hotspot["latitude"], hotspot["longitude"]),
        (hospital_lat, hospital_lon)
    ).km

    if distance < min_distance:

        min_distance = distance

        if pd.notna(hospital["name"]):
            nearest_hospital = hospital["name"]

print("Nearest Hospital:", nearest_hospital)
print("Distance:", round(min_distance, 2), "km")