import geopandas as gpd

hospitals = gpd.read_file("data/raw/hospitals.geojson")
industries = gpd.read_file("data/raw/industries.geojson")
factories = gpd.read_file("data/raw/factories.geojson")
powerplants = gpd.read_file("data/raw/powerplants.geojson")

print("Hospitals:", hospitals.shape)
print("Industries:", industries.shape)
print("Factories:", factories.shape)
print("Power Plants:", powerplants.shape)