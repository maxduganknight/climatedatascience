"""
Projection and coordinate transformation utilities.

Common projection operations used across the cdr_mapper project.
Follows DRY principles by centralizing CRS operations.
"""

import numpy as np
from pyproj import Transformer


def create_latitude_grid(
    raster_shape: tuple,
    raster_bounds: tuple,
    raster_crs: str,
) -> np.ndarray:
    """
    Create a 2D latitude grid for a raster in any CRS.

    Efficiently handles common projections (Mercator, WGS84) with optimized methods.
    Falls back to coordinate transformation for other projections.

    Parameters:
    -----------
    raster_shape : tuple
        (height, width) of the raster
    raster_bounds : tuple
        (left, bottom, right, top) bounds in raster CRS
    raster_crs : str
        CRS of the raster (e.g., "EPSG:3395")

    Returns:
    --------
    np.ndarray : 2D array of latitude values (shape: raster_shape)
    """
    height, width = raster_shape
    left, bottom, right, top = raster_bounds

    # Mercator projections: use inverse Mercator formula (fast and accurate)
    if "3395" in raster_crs or "3857" in raster_crs:
        # Web/World Mercator: lat = atan(sinh(y / R)) * 180 / pi
        R = 6378137.0  # Earth radius for both EPSG:3395 and EPSG:3857
        ys = np.linspace(top, bottom, height, dtype=np.float32)
        lats = np.degrees(np.arctan(np.sinh(ys / R)))
        lat_grid = np.tile(lats.reshape(-1, 1), (1, width))

    # Geographic coordinates: already in lat/lon
    elif "EPSG:4326" in raster_crs or "WGS84" in raster_crs:
        lats = np.linspace(top, bottom, height, dtype=np.float32)
        lat_grid = np.tile(lats.reshape(-1, 1), (1, width))

    # Other projections: transform coordinates
    else:
        transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

        # Get Y coordinates along the central meridian
        ys = np.linspace(top, bottom, height)
        center_x = (left + right) / 2

        # Transform in chunks to avoid memory issues
        chunk_size = 1000
        lats = np.zeros(height, dtype=np.float32)

        for i in range(0, height, chunk_size):
            end_i = min(i + chunk_size, height)
            chunk_ys = ys[i:end_i]
            chunk_xs = np.full_like(chunk_ys, center_x)

            # Transform this chunk
            _, chunk_lats = transformer.transform(chunk_xs, chunk_ys)
            lats[i:end_i] = chunk_lats

        # Broadcast to full grid
        lat_grid = np.tile(lats.reshape(-1, 1), (1, width))

    return lat_grid


def is_geographic_crs(crs: str) -> bool:
    """Check if a CRS is geographic (lat/lon) vs projected."""
    return "EPSG:4326" in crs or "WGS84" in crs


def is_mercator_crs(crs: str) -> bool:
    """Check if a CRS is Web or World Mercator."""
    return "3395" in crs or "3857" in crs


def get_pixel_size_meters(
    raster_bounds: tuple,
    raster_shape: tuple,
    raster_crs: str,
) -> float:
    """
    Calculate the approximate pixel size in meters.

    For projected CRS (meters), this is straightforward.
    For geographic CRS (degrees), uses equator approximation.

    Parameters:
    -----------
    raster_bounds : tuple
        (left, bottom, right, top) in raster CRS
    raster_shape : tuple
        (height, width)
    raster_crs : str
        CRS string

    Returns:
    --------
    float : Approximate pixel size in meters
    """
    left, bottom, right, top = raster_bounds
    height, width = raster_shape

    if is_geographic_crs(raster_crs):
        # Geographic: convert degrees to meters at equator
        # 1 degree ≈ 111.32 km at equator
        pixel_width_deg = (right - left) / width
        pixel_height_deg = (top - bottom) / height
        avg_pixel_deg = np.mean([pixel_width_deg, pixel_height_deg])
        return avg_pixel_deg * 111_320  # meters per degree at equator
    else:
        # Projected: already in meters (for most projections)
        pixel_width_meters = (right - left) / width
        pixel_height_meters = (top - bottom) / height
        return np.mean([pixel_width_meters, pixel_height_meters])


def clip_bounds_to_valid_mercator(
    bounds: tuple,
    crs: str,
) -> tuple:
    """
    Clip geographic bounds to valid Web Mercator range (±85.05°).

    Parameters:
    -----------
    bounds : tuple
        (left, bottom, right, top) in the given CRS
    crs : str
        CRS string

    Returns:
    --------
    tuple : Clipped bounds in same CRS
    """
    if not is_geographic_crs(crs):
        return bounds  # Only clip geographic coordinates

    MERCATOR_MAX_LAT = 85.05112878
    left, bottom, right, top = bounds

    bottom_clipped = max(bottom, -MERCATOR_MAX_LAT)
    top_clipped = min(top, MERCATOR_MAX_LAT)

    return (left, bottom_clipped, right, top_clipped)
