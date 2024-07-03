import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from scipy.ndimage import distance_transform_edt, generic_filter
import matplotlib.pyplot as plt

def model():
    # Load the vector data
    roads_path = "dataset/Roads.geojson"
    roads = gpd.read_file(roads_path)

    # Load the raster data
    elevation_path = "dataset/Elevation.tif"
    land_cover_path = "dataset/Land_Cover.tif"
    protected_status_path = "dataset/Protected_Status.tif"

    with rasterio.open(elevation_path) as src:
        elevation = src.read(1)
        profile = src.profile

    with rasterio.open(land_cover_path) as src:
        land_cover = src.read(1)

    with rasterio.open(protected_status_path) as src:
        protected_status = src.read(1)

    # Create a mask from the roads vector data
    roads_mask = geometry_mask(roads.geometry, transform=profile['transform'], invert=True, out_shape=(profile['height'], profile['width']))

    # Distance Accumulation (similar to DistanceAccumulation in ArcPy)
    distance_to_roads = distance_transform_edt(~roads_mask)

    # Save the distance raster
    distance_to_roads_path = "dataset/Mountain_Lion_Corridors/Distance_to_Roads.tif"
    with rasterio.open(distance_to_roads_path, 'w', **profile) as dst:
        dst.write(distance_to_roads, 1)

    # Display the distance raster
    plt.figure(figsize=(10, 8))
    plt.title("Distance to Roads")
    plt.imshow(distance_to_roads, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.show()

    # Focal Statistics (Range filter)
    def range_filter(values):
        return values.max() - values.min()

    ruggedness = generic_filter(elevation, range_filter, size=(3, 3))

    # Save the ruggedness raster
    ruggedness_path = "dataset/Mountain_Lion_Corridors/Ruggedness.tif"
    with rasterio.open(ruggedness_path, 'w', **profile) as dst:
        dst.write(ruggedness, 1)

    # Display the ruggedness raster
    plt.figure(figsize=(10, 8))
    plt.title("Ruggedness")
    plt.imshow(ruggedness, cmap='viridis')
    plt.colorbar(label='Ruggedness')
    plt.show()

    # Rescale by Function
    min_value, max_value = ruggedness.min(), ruggedness.max()
    ruggedness_cost = (ruggedness - min_value) / (max_value - min_value) * 10 + 1

    # Save the rescaled ruggedness raster
    ruggedness_cost_path = "dataset/Mountain_Lion_Corridors/Ruggedness_Cost.tif"
    with rasterio.open(ruggedness_cost_path, 'w', **profile) as dst:
        dst.write(ruggedness_cost, 1)

    # Display the rescaled ruggedness raster
    plt.figure(figsize=(10, 8))
    plt.title("Ruggedness Cost")
    plt.imshow(ruggedness_cost, cmap='viridis')
    plt.colorbar(label='Ruggedness Cost')
    plt.show()

    # Reclassify Land Cover
    land_cover_reclassification = {
        11: 10,  # Open Water
        21: 8,   # Developed, Open Space
        22: 7,   # Developed, Low Intensity
        23: 8,   # Developed, Medium Intensity
        24: 9,   # Developed, High Intensity
        31: 6,   # Barren Land
        41: 2,   # Deciduous Forest
        42: 1,   # Evergreen Forest
        43: 2,   # Mixed Forest
        52: 3,   # Shrub/Scrub
        71: 3,   # Herbaceous
        81: 4,   # Hay/Pasture
        82: 6,   # Cultivated Crops
        90: 4,   # Woody Wetlands
        95: 4,   # Emergent Herbaceous Wetlands
        255: 255 # NODATA
    }

    reclassified_land_cover = np.copy(land_cover)
    for old_value, new_value in land_cover_reclassification.items():
        reclassified_land_cover[land_cover == old_value] = new_value

    # Save the reclassified land cover raster
    reclassified_land_cover_path = "dataset/Mountain_Lion_Corridors/Reclassified_Land_Cover.tif"
    with rasterio.open(reclassified_land_cover_path, 'w', **profile) as dst:
        dst.write(reclassified_land_cover, 1)

    # Display the reclassified land cover raster
    plt.figure(figsize=(10, 8))
    plt.title("Reclassified Land Cover")
    plt.imshow(reclassified_land_cover, cmap='viridis')
    plt.colorbar(label='Land Cover Class')
    plt.show()

    # Reclassify Protected Status
    protected_status_reclassification = {
        1: 1,   # Very high
        2: 3,   # High
        3: 6,   # Medium
        4: 9,   # Not protected
        255: 10 # NODATA
    }

    reclassified_protected_status = np.copy(protected_status)
    for old_value, new_value in protected_status_reclassification.items():
        reclassified_protected_status[protected_status == old_value] = new_value

    # Save the reclassified protected status raster
    reclassified_protected_status_path = "dataset/Mountain_Lion_Corridors/Reclassified_Protected_Status.tif"
    with rasterio.open(reclassified_protected_status_path, 'w', **profile) as dst:
        dst.write(reclassified_protected_status, 1)

    # Display the reclassified protected status raster
    plt.figure(figsize=(10, 8))
    plt.title("Reclassified Protected Status")
    plt.imshow(reclassified_protected_status, cmap='viridis')
    plt.colorbar(label='Protected Status Class')
    plt.show()

    # Calculate weighted sum
    weight_ruggedness = 1.25
    weight_distance_to_roads = 1
    weight_land_cover = 1.25
    weight_protected_status = 1

    weighted_sum = (weight_ruggedness * ruggedness_cost +
                    weight_distance_to_roads * distance_to_roads +
                    weight_land_cover * reclassified_land_cover +
                    weight_protected_status * reclassified_protected_status)

    # Save the weighted sum raster
    weighted_sum_path = "dataset/Mountain_Lion_Corridors/Weighted_Sum.tif"
    with rasterio.open(weighted_sum_path, 'w', **profile) as dst:
        dst.write(weighted_sum, 1)

    # Save the weighted sum as a PNG image
    plt.figure(figsize=(10, 8))
    plt.title("Weighted Sum")
    plt.imshow(weighted_sum, cmap='viridis')
    plt.colorbar(label='Weighted Sum')
    plt.savefig("dataset/Mountain_Lion_Corridors/Weighted_Sum.png")
    plt.show()

if __name__ == "__main__":
    model()
