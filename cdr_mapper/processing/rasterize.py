"""
Rasterization utilities for converting vector geological data to raster format.

This reduces data size for web visualization by converting large vector datasets
to lightweight raster grids.

IMPORTANT - Projection Requirements:
====================================
Web maps (including Folium) use Web Mercator projection (EPSG:3857) for base map tiles.
To ensure proper alignment between our data overlays and the base map:

1. Vector data is rasterized in EPSG:4326 (WGS84 lat/lon)
2. The raster is then reprojected to Web Mercator (EPSG:3857)
3. Latitude is clipped to ±85.05° (Web Mercator's valid range)
4. The reprojected raster is displayed on the Web Mercator base map

DO NOT:
- Use EPSG:4326 for the base map (causes tile misalignment)
- Skip the Web Mercator reprojection step (causes vertical stretching)
- Flip arrays vertically (not needed with proper reprojection)

The reprojection corrects for Web Mercator's vertical distortion that increases
with distance from the equator. Without this, geological features appear displaced:
- Iceland data appears north of Siberia
- Australia data appears in the wrong ocean
- Only equatorial regions align correctly
"""

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject


def rasterize_geological_layer(
    gdf: gpd.GeoDataFrame,
    resolution: float = 0.5,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Rasterize geological vector data to a boolean grid.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Geological vector data to rasterize
    resolution : float
        Grid resolution in degrees (default: 0.5°)
    bounds : tuple, optional
        (lon_min, lat_min, lon_max, lat_max). If None, uses global extent
    cache_path : Path, optional
        Path to save cached raster as .npz

    Returns:
    --------
    tuple: (raster_array, extent) where:
        - raster_array: boolean numpy array (True where geology present)
        - extent: (lon_min, lon_max, lat_min, lat_max)
    """
    # Check cache first
    if cache_path is not None and cache_path.exists():
        print(f"  Loading rasterized data from cache: {cache_path}", flush=True)
        cached = np.load(cache_path)
        return cached["data"], tuple(cached["extent"])

    print(f"  Rasterizing vector data at {resolution}° resolution...", flush=True)

    # Set bounds
    if bounds is None:
        bounds = (-180, -90, 180, 90)  # Global

    lon_min, lat_min, lon_max, lat_max = bounds

    # Calculate grid dimensions
    width = int((lon_max - lon_min) / resolution)
    height = int((lat_max - lat_min) / resolution)

    print(f"  Grid size: {width} x {height} ({width * height:,} pixels)", flush=True)

    # Create affine transform
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    # Rasterize geometries
    # Value of 1 where geology is present, 0 elsewhere
    shapes = ((geom, 1) for geom in gdf.geometry)
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,  # Include pixels touched by geometry
    )

    # Convert to boolean for memory efficiency
    raster = raster.astype(bool)

    coverage = (raster.sum() / raster.size) * 100
    print(f"  Coverage: {coverage:.2f}% of grid", flush=True)

    # Save to cache if specified
    if cache_path is not None:
        print(f"  Saving rasterized data to cache: {cache_path}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            data=raster,
            extent=np.array([lon_min, lat_min, lon_max, lat_max]),
        )

    # Return extent in standard format: (lon_min, lat_min, lon_max, lat_max)
    # This matches typical GIS convention (west, south, east, north)
    extent = (lon_min, lat_min, lon_max, lat_max)
    return raster, extent


def raster_to_image_overlay(
    raster: np.ndarray,
    extent: Tuple[float, float, float, float],
    color: str = "#C94545",
    opacity: float = 0.6,
    smooth_edges: bool = True,
) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert boolean raster to RGBA image for folium ImageOverlay.

    Parameters:
    -----------
    raster : ndarray
        Boolean raster array
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max)
    color : str
        Hex color code
    opacity : float
        Opacity (0-1)
    smooth_edges : bool
        Apply edge smoothing to reduce blocky appearance (default: True)

    Returns:
    --------
    tuple: (rgba_image, bounds) where:
        - rgba_image: RGBA numpy array (height, width, 4)
        - bounds: ((lat_min, lon_min), (lat_max, lon_max)) for folium
    """
    from scipy.ndimage import gaussian_filter

    # Parse hex color
    color = color.lstrip("#")
    r, g, b = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))

    # Create RGBA image
    height, width = raster.shape

    # Apply light Gaussian smoothing to edges for anti-aliasing effect
    if smooth_edges:
        # Convert bool to float for smoothing
        raster_float = raster.astype(np.float32)
        # Very light smoothing (sigma=0.5) to soften edges without blurring too much
        raster_smooth = gaussian_filter(raster_float, sigma=0.5)
        # Use smoothed values for alpha channel to create soft edges
        alpha_values = (raster_smooth * opacity * 255).astype(np.uint8)
    else:
        alpha_values = (raster.astype(np.float32) * opacity * 255).astype(np.uint8)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Set color where raster is True (use any pixel with alpha > 0)
    has_color = alpha_values > 0
    rgba[has_color, 0] = r  # Red
    rgba[has_color, 1] = g  # Green
    rgba[has_color, 2] = b  # Blue
    rgba[:, :, 3] = alpha_values  # Alpha with smoothed edges

    # Convert extent to Folium bounds format
    # Folium expects [[south_lat, west_lon], [north_lat, east_lon]]
    # Convert numpy types to native Python types for JSON serialization
    lon_min, lat_min, lon_max, lat_max = extent
    bounds = [
        [float(lat_min), float(lon_min)],  # Southwest corner
        [float(lat_max), float(lon_max)],  # Northeast corner
    ]

    return rgba, bounds


def reproject_raster_to_web_mercator(
    raster: np.ndarray,
    extent: Tuple[float, float, float, float],
    is_boolean: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Reproject a raster from EPSG:4326 to Web Mercator (EPSG:3857).

    Parameters:
    -----------
    raster : ndarray
        Raster array in EPSG:4326 (boolean or float)
    extent : tuple
        (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
    is_boolean : bool
        If True, treats as boolean mask. If False, preserves float values (e.g., thickness data)

    Returns:
    --------
    tuple: (reprojected_raster, mercator_extent) where:
        - reprojected_raster: array in Web Mercator projection (bool or float32)
        - mercator_extent: (lon_min, lat_min, lon_max, lat_max) in EPSG:4326 for Folium
    """
    print("  Reprojecting raster to Web Mercator (EPSG:3857)...", flush=True)

    lon_min, lat_min, lon_max, lat_max = extent
    height, width = raster.shape

    # Clip to valid Web Mercator latitude range
    MERCATOR_MAX_LAT = 85.05112878
    lat_min_clipped = max(lat_min, -MERCATOR_MAX_LAT)
    lat_max_clipped = min(lat_max, MERCATOR_MAX_LAT)

    if lat_min_clipped != lat_min or lat_max_clipped != lat_max:
        print(
            f"  Clipping latitude from [{lat_min}, {lat_max}] to [{lat_min_clipped}, {lat_max_clipped}]",
            flush=True,
        )

        # Crop the raster to the valid latitude range
        pixel_height = (lat_max - lat_min) / height

        # Calculate which rows to keep
        row_min = int((lat_max - lat_max_clipped) / pixel_height)
        row_max = height - int((lat_min_clipped - lat_min) / pixel_height)

        raster = raster[row_min:row_max, :]
        lat_min, lat_max = lat_min_clipped, lat_max_clipped
        height = raster.shape[0]

    # Create source transform (EPSG:4326)
    src_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)

    # Calculate destination transform and dimensions
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        left=lon_min,
        bottom=lat_min,
        right=lon_max,
        top=lat_max,
    )

    # Create destination array with appropriate dtype
    if is_boolean:
        dst_raster = np.zeros((dst_height, dst_width), dtype=np.uint8)
        src_raster = raster.astype(np.uint8)
        resampling_method = Resampling.nearest
        src_nodata = None
        dst_nodata = None
    else:
        dst_raster = np.zeros((dst_height, dst_width), dtype=np.float32)
        src_raster = raster.astype(np.float32)
        resampling_method = Resampling.bilinear  # Better for continuous data
        src_nodata = np.nan
        dst_nodata = np.nan

    # Reproject
    reproject(
        source=src_raster,
        destination=dst_raster,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling_method,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )

    # Convert to boolean if needed
    if is_boolean:
        dst_raster = dst_raster.astype(bool)

    # Return the extent in lat/lon (EPSG:4326) format since Folium expects that
    # even for Web Mercator overlays
    mercator_extent = (lon_min, lat_min, lon_max, lat_max)

    print(f"  Reprojected: {width}x{height} -> {dst_width}x{dst_height}", flush=True)
    print(
        f"  Extent (lat/lon): lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]",
        flush=True,
    )

    return dst_raster, mercator_extent
