import os
import geopandas as gpd
import numpy as np


def get_utm_epsg(gdf):
    """Estimate UTM EPSG code based on the centroid of the input GeoDataFrame."""
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    centroid = gdf_wgs84.geometry.union_all().centroid
    lon = centroid.x
    utm_zone = int(np.ceil((lon + 180) / 6))
    epsg_code = 32600 + utm_zone
    return epsg_code


def load_points_information(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    points_gdf = gpd.read_file(path)
    if points_gdf.empty:
        raise ValueError("Input GeoDataFrame is empty!")
    print(f"Points CRS: {points_gdf.crs}")
    utm_epsg = get_utm_epsg(points_gdf)
    points_gdf = points_gdf.to_crs(epsg=utm_epsg)
    print(f"Points convert to the UTM：EPSG:{utm_epsg}")
    print(f"Number of point: {len(points_gdf)}")
    return points_gdf
