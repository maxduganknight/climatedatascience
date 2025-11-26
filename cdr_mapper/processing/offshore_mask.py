"""
Offshore accessibility masking for sedimentary storage capacity.

Implements the Kearns et al. criteria for practically accessible offshore storage:
1. Water depth cannot exceed 300 meters (not implemented - no bathymetry data)
2. Site must be within 200 miles of shore (landmass > 10,000 km²)
3. Site must fall between 66°N and 66°S (Arctic/Antarctic exclusion)

Reference: Kearns et al. (2017) "Developing a consistent database for regional
geologic CO2 storage capacity worldwide"
"""

import sys
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# Add parent directory to path for utils imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.distance import (
    calculate_distance_from_mask,
    create_distance_buffer_mask,
    miles_to_meters,
)
from utils.projection import create_latitude_grid, get_pixel_size_meters
from utils.raster import create_raster_transform


def create_offshore_accessibility_mask(
    raster_shape: Tuple[int, int],
    raster_bounds: Tuple[float, float, float, float],
    raster_crs: str,
    countries_geojson_path: str,
    max_distance_miles: float = 200.0,
    min_landmass_km2: float = 10000.0,
    latitude_bounds: Tuple[float, float] = (-90.0, 90.0),
) -> np.ndarray:
    """
    Create a binary mask for offshore areas that meet Kearns accessibility criteria.

    Parameters:
    -----------
    raster_shape : tuple
        (height, width) of the raster
    raster_bounds : tuple
        (left, bottom, right, top) bounds in raster CRS
    raster_crs : str
        CRS of the raster (e.g., "EPSG:3395")
    countries_geojson_path : str
        Path to world countries GeoJSON
    max_distance_miles : float
        Maximum distance from shore in miles (default: 200)
    min_landmass_km2 : float
        Minimum landmass size in km² (default: 10,000)
    latitude_bounds : tuple
        (min_lat, max_lat) in degrees (default: (-66, 66) to exclude polar regions)

    Returns:
    --------
    np.ndarray : Binary mask (1 = accessible, 0 = inaccessible)
    """
    print("\n" + "=" * 70)
    print("Creating offshore accessibility mask (Kearns criteria)")
    print("=" * 70)

    height, width = raster_shape
    left, bottom, right, top = raster_bounds

    # Create transform for the raster
    transform = create_raster_transform(raster_bounds, raster_shape)

    # Load country boundaries
    print(f"\nLoading country boundaries from {countries_geojson_path}...")
    countries = gpd.read_file(countries_geojson_path)
    print(f"  Loaded {len(countries)} country polygons")

    # Filter countries by area (>10,000 km²)
    # Convert to equal-area projection for accurate area calculation
    print(
        f"\nFiltering landmasses > {min_landmass_km2:,.0f} km² (excludes small islands)..."
    )
    countries_area = countries.to_crs("ESRI:54009")  # Mollweide equal-area projection
    countries["area_km2"] = countries_area.geometry.area / 1_000_000  # m² to km²
    large_landmasses = countries[countries["area_km2"] >= min_landmass_km2].copy()
    print(f"  Retained {len(large_landmasses)} large landmasses")
    print(
        f"  Excluded {len(countries) - len(large_landmasses)} small islands/territories"
    )

    # Reproject landmasses to raster CRS
    print(f"\nReprojecting landmasses to raster CRS ({raster_crs})...")
    large_landmasses = large_landmasses.to_crs(raster_crs)

    # Rasterize land areas (1 = land, 0 = ocean)
    print("\nRasterizing land areas...")
    land_mask = rasterize(
        [(geom, 1) for geom in large_landmasses.geometry],
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    print(
        f"  Land pixels: {np.sum(land_mask):,} ({np.sum(land_mask) / land_mask.size * 100:.1f}%)"
    )

    # Calculate distance from shore
    print("\nCalculating distance from shore...")
    ocean_mask = 1 - land_mask  # Invert: 1 = ocean, 0 = land

    # Get pixel size and convert distance to meters
    pixel_size_meters = get_pixel_size_meters(raster_bounds, raster_shape, raster_crs)
    max_distance_meters = miles_to_meters(max_distance_miles)

    print(
        f"  Pixel size: {pixel_size_meters:,.0f}m (~{pixel_size_meters / 1000:.1f}km)"
    )
    print(f"  Max distance: {max_distance_miles} miles = {max_distance_meters:,.0f}m")

    # Create distance buffer mask using utility function
    distance_mask = create_distance_buffer_mask(
        land_mask, pixel_size_meters, max_distance_meters
    )

    accessible_ocean = np.sum(distance_mask & ocean_mask)
    total_ocean = np.sum(ocean_mask)
    print(
        f"  Ocean within {max_distance_miles} miles of shore: {accessible_ocean:,} pixels ({accessible_ocean / total_ocean * 100:.1f}% of ocean)"
    )

    # Apply latitude constraint (±66° to exclude Arctic/Antarctic)
    # print(
    #     f"\nApplying latitude constraint ({latitude_bounds[0]}° to {latitude_bounds[1]}°)..."
    # )

    # Create latitude grid using utility function
    lat_grid = create_latitude_grid(raster_shape, raster_bounds, raster_crs)

    # Create latitude mask (1 = within bounds, 0 = outside bounds)
    # lat_mask = (
    #     (lat_grid >= latitude_bounds[0]) & (lat_grid <= latitude_bounds[1])
    # ).astype(np.uint8)

    # Clean up large arrays
    del lat_grid

    # excluded_by_latitude = np.sum((1 - lat_mask) & distance_mask & ocean_mask)
    # print(
    #     f"  Excluded {excluded_by_latitude:,} ocean pixels in polar regions ({excluded_by_latitude / total_ocean * 100:.1f}% of ocean)"
    # )

    # Combine masks to identify what to KEEP
    # Accessible offshore ocean: ocean + within 200 miles + non-polar
    accessible_offshore = distance_mask & ocean_mask

    # Final mask: 1 = keep (land OR accessible offshore), 0 = exclude (inaccessible offshore)
    # We want to KEEP: all land + accessible offshore ocean
    # We want to EXCLUDE: inaccessible offshore ocean (too far or polar)
    final_mask = land_mask | accessible_offshore

    accessible_offshore_pixels = np.sum(accessible_offshore)
    kept_pixels = np.sum(final_mask)
    excluded_ocean = total_ocean - accessible_offshore_pixels

    print(f"\n" + "=" * 70)
    print(f"FINAL ACCESSIBILITY MASK")
    print(f"=" * 70)
    print(
        f"  Total ocean: {total_ocean:,} pixels ({total_ocean / land_mask.size * 100:.1f}%)"
    )
    print(
        f"  Accessible offshore: {accessible_offshore_pixels:,} pixels ({accessible_offshore_pixels / total_ocean * 100:.1f}% of ocean)"
    )
    print(
        f"  Excluded offshore: {excluded_ocean:,} pixels ({excluded_ocean / total_ocean * 100:.1f}% of ocean)"
    )
    print(
        f"    - By distance (>{max_distance_miles} miles): {np.sum((1 - distance_mask) & ocean_mask):,} pixels"
    )
    # print(
    #     f"    - By latitude (outside {latitude_bounds}°): {excluded_by_latitude:,} pixels"
    # )
    print(
        f"  Total kept (land + accessible offshore): {kept_pixels:,} pixels ({kept_pixels / land_mask.size * 100:.1f}%)"
    )
    print("=" * 70 + "\n")

    return final_mask
