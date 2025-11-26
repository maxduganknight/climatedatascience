"""
Geothermal data interpolation utilities
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def interpolate_geothermal(
    df: pd.DataFrame,
    extent: tuple,
    target_shape: tuple,
    grid_resolution: float = 2.0,
    smoothing_sigma: float = 0.5,
) -> np.ndarray:
    """
    Interpolate geothermal point data to regular grid.

    Parameters:
    -----------
    df : DataFrame
        Geothermal data with columns 'long_EW', 'lat_NS', 'q'
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max)
    target_shape : tuple
        (height, width) of output grid
    grid_resolution : float
        Bin size in degrees (coarser = smoother)
    smoothing_sigma : float
        Gaussian smoothing sigma

    Returns:
    --------
    ndarray: Interpolated heat flow values (mW/m²)
    """
    lon_min, lon_max, lat_min, lat_max = extent
    height, width = target_shape

    # Create output grid
    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)  # Reversed for image coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Bin the data to reduce noise
    lon_bins = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
    lat_bins = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)

    df = df.copy()
    df["lon_bin"] = pd.cut(df["long_EW"], bins=lon_bins, labels=False)
    df["lat_bin"] = pd.cut(df["lat_NS"], bins=lat_bins, labels=False)

    # Average within bins
    binned = df.groupby(["lon_bin", "lat_bin"])["q"].mean().reset_index()
    binned["long_EW"] = lon_bins[binned["lon_bin"].astype(int)] + grid_resolution / 2
    binned["lat_NS"] = lat_bins[binned["lat_bin"].astype(int)] + grid_resolution / 2

    # Interpolate using griddata
    points = binned[["long_EW", "lat_NS"]].values
    values = binned["q"].values

    geo_grid = griddata(
        points, values, (lon_grid, lat_grid), method="linear", fill_value=np.nan
    )

    # Fill gaps with nearest neighbor
    mask = np.isnan(geo_grid)
    if mask.any():
        geo_grid_nn = griddata(points, values, (lon_grid, lat_grid), method="nearest")
        geo_grid[mask] = geo_grid_nn[mask]

    # Apply smoothing
    geo_grid = gaussian_filter(geo_grid, sigma=smoothing_sigma)

    # Mask unrealistic values
    geo_grid = np.ma.masked_where((geo_grid < 0) | (geo_grid > 300), geo_grid)

    return geo_grid


def heat_flow_to_depth(
    heat_flow: np.ndarray,
    target_temp: float = 150,
    surface_temp: float = 15,
    thermal_conductivity: float = 2.5,
) -> np.ndarray:
    """
    Convert geothermal heat flow to drilling depth required to reach target temperature.

    Parameters:
    -----------
    heat_flow : ndarray
        Heat flow in mW/m²
    target_temp : float
        Target temperature in °C (default: 150)
    surface_temp : float
        Surface temperature in °C (default: 15)
    thermal_conductivity : float
        Thermal conductivity in W/m·K (default: 2.5)

    Returns:
    --------
    ndarray: Depth in km to reach target temperature
    """
    geothermal_gradient = heat_flow / thermal_conductivity
    depth_km = (target_temp - surface_temp) / geothermal_gradient
    return depth_km
