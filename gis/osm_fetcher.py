# import osmnx as ox

# place = "Hyderabad, Telangana, India"

# tags = {
#     "amenity": "hospital"
# }

# hospitals = ox.features_from_place(
#     place,
#     tags
# )

# print(hospitals.head())

# hospitals.to_file(
#     "data/raw/hospitals.geojson",
#     driver="GeoJSON"
# )


# print("Hospitals saved successfully!")

# import osmnx as ox

# place = "Hyderabad, Telangana, India"

# industries = ox.features_from_place(
#     place,
#     {"landuse": "industrial"}
# )

# industries.to_file(
#     "data/raw/industries.geojson",
#     driver="GeoJSON"
# )

# print("Industries saved")



# import osmnx as ox

# place = "Hyderabad, Telangana, India"
# industries = ox.features_from_place(
#     place,
#     {"landuse": "industrial"}
# )

# print(industries.shape)

# industries.to_file(
#     "data/raw/refineries.geojson",
#     driver="GeoJSON"
# )

# print("Refineries saved")


# powerplants = ox.features_from_place(
#     place,
#     {"power": "plant"}
# )

# powerplants.to_file(
#     "data/raw/powerplants.geojson",
#     driver="GeoJSON"
# )
# print("Factories saved")

import osmnx as ox

place = "Telangana, India"

refineries = ox.features_from_place(
    place,
    {"industrial": "refinery"}
)

print(refineries.shape)

refineries.to_file(
    "data/raw/refineries.geojson",
    driver="GeoJSON"
)
print("Refineries saved")