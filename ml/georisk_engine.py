import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic

# Load datasets

hotspots = pd.read_csv(
    "data/processed/hotspots_with_risk.csv"
)

hospitals = gpd.read_file(
    "data/raw/hospitals.geojson"
)

industries = gpd.read_file(
    "data/raw/industries.geojson"
)

factories = gpd.read_file(
    "data/raw/factories.geojson"
)

powerplants = gpd.read_file(
    "data/raw/powerplants.geojson"
)

print("Hotspots:", hotspots.shape)
print("Hospitals:", hospitals.shape)
print("Industries:", industries.shape)
print("Factories:", factories.shape)
print("Power Plants:", powerplants.shape)


def nearest_distance(lat, lon, gdf):

    min_distance = float("inf")

    for _, row in gdf.iterrows():

        point = row.geometry.centroid

        distance = geodesic(
            (lat, lon),
            (point.y, point.x)
        ).km

        if distance < min_distance:
            min_distance = distance

    return round(min_distance, 2)


# Apply on ALL hotspots

hotspots["hospital_km"] = hotspots.apply(
    lambda x: nearest_distance(
        x["latitude"],
        x["longitude"],
        hospitals
    ),
    axis=1
)

hotspots["industry_km"] = hotspots.apply(
    lambda x: nearest_distance(
        x["latitude"],
        x["longitude"],
        industries
    ),
    axis=1
)

hotspots["factory_km"] = hotspots.apply(
    lambda x: nearest_distance(
        x["latitude"],
        x["longitude"],
        factories
    ),
    axis=1
)

hotspots["powerplant_km"] = hotspots.apply(
    lambda x: nearest_distance(
        x["latitude"],
        x["longitude"],
        powerplants
    ),
    axis=1
)


def calculate_georisk(
    risk_score,
    hospital_km,
    industry_km,
    factory_km,
    powerplant_km
):

    score = (
        risk_score
        + (hospital_km / 100)
        + (industry_km / 100)
        + (factory_km / 100)
        + (powerplant_km / 100)
    )

    return round(score, 2)


hotspots["georisk_score"] = hotspots.apply(
    lambda x: calculate_georisk(
        x["risk_score"],
        x["hospital_km"],
        x["industry_km"],
        x["factory_km"],
        x["powerplant_km"]
    ),
    axis=1
)


def georisk_level(score):

    if score >= 120:
        return "Critical"

    elif score >= 80:
        return "High"

    elif score >= 50:
        return "Medium"

    else:
        return "Low"


hotspots["georisk_level"] = hotspots[
    "georisk_score"
].apply(georisk_level)

print(
    hotspots[
        [
            "risk_score",
            "georisk_score",
            "georisk_level"
        ]
    ].head()
)

print(
    hotspots["georisk_level"].value_counts()
)

hotspots.to_csv(
    "data/processed/georisk_hotspots.csv",
    index=False
)

print("GeoRisk dataset saved successfully!")