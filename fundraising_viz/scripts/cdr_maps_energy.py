"""
Energy Potential Map for CDR Operations

This script creates a visualization of global energy potential for powering CDR operations
based on solar photovoltaic potential and enhanced geothermal (heat flow) potential.

Data sources:
1. Solar PV: Global Solar Atlas PVOUT data (geotiff format)
2. Geothermal: IHFC Global Heat Flow Database (point data requiring interpolation)
"""

import os
import sys

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling
from scipy.ndimage import gaussian_filter

# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(parent_dir), "reports"))

from map_utils import (
    MAP_STYLE,
    REGIONS,
    WORLD_COUNTRIES_GEOJSON,
    add_base_features,
    add_deepsky_icon,
    create_land_mask,
    create_legend_item,
    load_countries_from_geojson,
    load_country_shapefile,
    sanitize_region_name,
    save_map,
)
from utils import COLORS, setup_space_mono_font

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# ENERGY POTENTIAL ASSUMPTIONS
# =============================================================================
# These assumptions define parameters for calculating regional energy potential
# in TWh/year for solar PV and enhanced geothermal systems
ENERGY_ASSUMPTIONS = {
    "conservative": {
        # Solar
        "solar_land_availability": 0.005,  # 1% of suitable land
        "solar_power_density_MW_km2": 24,  # MW/km²
        "solar_capacity_factor": 0.18,  # 18% (moderate climates)
        "solar_system_efficiency": 0.80,  # 85% DC-to-AC
        "min_solar_threshold": 2,
        # Geothermal
        "geo_max_depth_km": 4,  # Only proven commercial depths
        "geo_well_spacing_km2": 5,  # 5 km² per well
        "geo_well_power_MW": 5,  # 5 MW per well
        "geo_capacity_factor": 0.90,  # 90%
        "geo_land_eligibility": 0.15,  # 15% land eligible (Franzmann: conservative regional assumption)
    },
    "moderate": {
        # Solar
        "solar_land_availability": 0.0275,  # 2.5% of suitable land
        "solar_power_density_MW_km2": 35,  # MW/km²
        "solar_capacity_factor": 0.22,  # 22% (good climates)
        "solar_system_efficiency": 0.85,
        "min_solar_threshold": 3,
        # Geothermal
        "geo_max_depth_km": 6,  # Proven EGS technology
        "geo_well_spacing_km2": 3,  # 3 km² per well
        "geo_well_power_MW": 7.5,  # 7.5 MW per well
        "geo_capacity_factor": 0.92,  # 92%
        "geo_land_eligibility": 0.25,  # 25% land eligible (Franzmann: global average)
    },
    "optimistic": {
        # Solar
        "solar_land_availability": 0.05,  # 5% of suitable land
        "solar_power_density_MW_km2": 40,  # MW/km²
        "solar_capacity_factor": 0.26,  # 26% (excellent climates)
        "solar_system_efficiency": 0.90,
        "min_solar_threshold": 4,
        # Geothermal
        "geo_max_depth_km": 8,  # Modern drilling capability
        "geo_well_spacing_km2": 2,  # 2 km² per well
        "geo_well_power_MW": 10,  # 10 MW per well
        "geo_capacity_factor": 0.95,  # 95%
        "geo_land_eligibility": 0.40,  # 40% land eligible (Franzmann: high eligibility regions)
    },
    "miyake_equivalent": {
        "solar_land_availability": 0.15,  # KEY CHANGE
        "solar_power_density_MW_km2": 25,  # Match Miyake
        "solar_capacity_factor": 0.125,  # Match Miyake (1,100 hrs/yr)
        "solar_system_efficiency": 1.0,
        "min_solar_threshold": 3.7,
        # placeholder geothermal values
        "geo_max_depth_km": 8,
        "geo_well_spacing_km2": 2,
        "geo_well_power_MW": 10,
        "geo_capacity_factor": 0.95,
        "geo_land_eligibility": 0.25,  # 25% land eligible (Franzmann: global average)
    },
}

# =============================================================================
# SOLAR PV THRESHOLDS AND COLORS (kWh/kWp/day)
# =============================================================================
# These thresholds define the classification levels for solar photovoltaic potential
# Adjust these values to make the maps more or less selective
# Lower thresholds show more widespread solar potential
SOLAR_THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 6.0]  # kWh/kWp/day

# Corresponding colors for each threshold level (light to dark amber/golden)
SOLAR_COLORS = [
    "#FBE9C8",  # Very light golden - 1.5-2.5 kWh/kWp/day
    "#F4D6A3",  # Light golden - 2.5-3.5 kWh/kWp/day
    "#E8A33D",  # Medium amber - 3.5-4.5 kWh/kWp/day
    "#C27D1B",  # Dark golden brown - 4.5-5.5 kWh/kWp/day
    "#8B5A0F",  # Very dark brown/bronze - ≥5.5 kWh/kWp/day
]

# Solar threshold descriptions for legends
SOLAR_THRESHOLD_LABELS = [
    (
        f"{SOLAR_THRESHOLDS[0]}-{SOLAR_THRESHOLDS[1]} kWh/kWp/day",
        "Viable with modern panels",
    ),
    (
        f"{SOLAR_THRESHOLDS[1]}-{SOLAR_THRESHOLDS[2]} kWh/kWp/day",
        "Moderate solar resource",
    ),
    (f"{SOLAR_THRESHOLDS[2]}-{SOLAR_THRESHOLDS[3]} kWh/kWp/day", "Good solar resource"),
    (
        f"{SOLAR_THRESHOLDS[3]}-{SOLAR_THRESHOLDS[4]} kWh/kWp/day",
        "Strong solar resource",
    ),
    (f"≥{SOLAR_THRESHOLDS[4]} kWh/kWp/day", "Excellent solar resource"),
]

# =============================================================================
# GEOTHERMAL THRESHOLDS AND PARAMETERS
# =============================================================================
# Parameters for converting heat flow to drilling depth
GEOTHERMAL_TARGET_TEMP = 150  # °C - target temperature for EGS
GEOTHERMAL_SURFACE_TEMP = 15  # °C - average surface temperature
GEOTHERMAL_THERMAL_CONDUCTIVITY = 2.5  # W/m·K - average continental crust

# Depth thresholds for enhanced geothermal systems (km)
# Shallower = easier to drill = better resource
GEOTHERMAL_DEPTH_THRESHOLDS = [4, 6, 8]  # km

# Geothermal gradient colors (dark to light red - darker = shallower = better)
GEOTHERMAL_COLORS = [
    "#8B2323",  # Dark burgundy - Excellent (≤4 km)
    "#C94545",  # Medium burgundy - Good (4-6 km)
    "#DE8A80",  # Light coral - Moderate (6-8 km)
    "#F2C4BE",  # Very light coral - Challenging (>8 km)
]

# Geothermal depth labels for legends
GEOTHERMAL_DEPTH_LABELS = [
    ("≤4 km depth", "Excellent - commercially viable today"),
    ("4-6 km depth", "Good - proven EGS technology"),
    ("6-8 km depth", "Moderate - feasible with modern drilling"),
    (">8 km depth", "Challenging - requires advanced tech"),
]


def heat_flow_to_depth(heat_flow_mw_m2):
    """
    Convert geothermal heat flow to drilling depth required to reach target temperature.

    Uses global constants:
    - GEOTHERMAL_TARGET_TEMP: Target temperature for EGS (°C)
    - GEOTHERMAL_SURFACE_TEMP: Average surface temperature (°C)
    - GEOTHERMAL_THERMAL_CONDUCTIVITY: Thermal conductivity of crust (W/m·K)

    Parameters:
    -----------
    heat_flow_mw_m2 : float or ndarray
        Heat flow in mW/m²

    Returns:
    --------
    float or ndarray: Depth in km to reach target temperature

    Formula:
    --------
    Depth (km) = (T_target - T_surface) / Geothermal_Gradient
    where Geothermal_Gradient (°C/km) = Heat_Flow (mW/m²) / Thermal_Conductivity (W/m·K)
    """
    geothermal_gradient = heat_flow_mw_m2 / GEOTHERMAL_THERMAL_CONDUCTIVITY
    depth_km = (GEOTHERMAL_TARGET_TEMP - GEOTHERMAL_SURFACE_TEMP) / geothermal_gradient
    return depth_km


def load_solar_data(
    geotiff_path="../data/cdr_mapper/energy/solar/World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/PVOUT.tif",
    target_resolution=0.5,
):
    """
    Load and downsample solar PV potential data from Global Solar Atlas.

    Parameters:
    -----------
    geotiff_path : str
        Path to the PVOUT geotiff file
    target_resolution : float
        Target resolution in degrees (default 0.5° ≈ 55km at equator)

    Returns:
    --------
    tuple: (data_array, extent) where extent is (lon_min, lon_max, lat_min, lat_max)
    """
    print("=" * 60)
    print("Loading solar PV potential data...")
    print("=" * 60)

    with rasterio.open(geotiff_path) as src:
        print(f"  Original shape: {src.shape}")
        print(f"  Original resolution: {src.res}")
        print(f"  Bounds: {src.bounds}")

        # Calculate downsampling factor
        downsample_factor = int(target_resolution / src.res[0])
        print(f"  Downsample factor: {downsample_factor}")

        # Read and downsample using windowed reading for memory efficiency
        data = src.read(
            1,
            out_shape=(src.height // downsample_factor, src.width // downsample_factor),
            resampling=Resampling.average,
        )

        # Get the new transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )

        # Calculate extent
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # Replace no-data values with NaN
    data = data.astype(float)
    data[data <= 0] = np.nan
    data[data > 10000] = np.nan

    print(f"  Final shape: {data.shape}")
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        print(
            f"  Value range: {valid_data.min():.1f} - {valid_data.max():.1f} kWh/kWp/day"
        )
    else:
        print(f"  Warning: No valid data found!")
    print("  ✓ Solar data loaded")

    return data, extent


def load_geothermal_data(
    excel_path="../data/cdr_mapper/energy/geothermal/IHFC_global/IHFC_2024_GHFDB.xlsx",
):
    """
    Load geothermal heat flow point data from IHFC Global Heat Flow Database.

    Parameters:
    -----------
    excel_path : str
        Path to the IHFC Excel file

    Returns:
    --------
    GeoDataFrame with heat flow points
    """
    print("\n" + "=" * 60)
    print("Loading geothermal heat flow data...")
    print("=" * 60)

    # Load with first row as header
    df = pd.read_excel(excel_path, skiprows=4, nrows=1)
    col_names = df.iloc[0].tolist()

    # Load the actual data
    df_data = pd.read_excel(excel_path, skiprows=5)
    df_data.columns = col_names

    # Convert to numeric
    df_data["q"] = pd.to_numeric(df_data["q"], errors="coerce")
    df_data["lat_NS"] = pd.to_numeric(df_data["lat_NS"], errors="coerce")
    df_data["long_EW"] = pd.to_numeric(df_data["long_EW"], errors="coerce")

    # Filter valid data
    df_clean = df_data[["q", "lat_NS", "long_EW"]].dropna()

    # Remove extreme outliers (likely measurement errors)
    # Keep values between 0 and 500 mW/m² (reasonable geothermal range)
    df_clean = df_clean[(df_clean["q"] > 0) & (df_clean["q"] < 500)]

    print(f"  Loaded {len(df_clean):,} valid measurements")
    print(
        f"  Heat flow range: {df_clean['q'].min():.1f} - {df_clean['q'].max():.1f} mW/m²"
    )
    print("  ✓ Geothermal data loaded")

    return df_clean


def interpolate_geothermal(
    df_geo,
    extent,
    shape,
    grid_resolution=1,
    smoothing_sigma=0.5,
    use_full_latitude=False,
):
    """
    Interpolate geothermal point data to grid using griddata

    Parameters:
    -----------
    df_geo : DataFrame
        Geothermal point data with columns 'long_EW', 'lat_NS', 'q'
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max)
    shape : tuple
        (height, width) of output grid
    grid_resolution : int
        Grid cell size in degrees for binning (higher = coarser = faster)
    use_full_latitude : bool
        If True, override extent to use full -90 to 90 latitude range

    Returns:
    --------
    2D numpy array with interpolated heat flow values
    """
    from scipy.interpolate import griddata

    print("\n" + "=" * 60)
    print("Interpolating geothermal data...")
    print("=" * 60)

    # Create output grid
    lon_min, lon_max, lat_min, lat_max = extent

    # Override latitude range if requested
    if use_full_latitude:
        lat_min = -90.0
        lat_max = 90.0
        # Adjust height to maintain similar resolution
        lat_range_original = extent[3] - extent[2]  # top - bottom
        height = int(shape[0] * (180.0 / lat_range_original))
        print(f"  Using FULL latitude range: -90° to 90° (adjusted height: {height})")

    width = shape[1]

    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)  # Note: reversed for image coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    print(f"  Interpolating to grid: {shape}")
    print(f"  Using {len(df_geo):,} point measurements")

    # Strategy 1: Bin the data into grid cells and average (much faster)
    print(f"  Binning data into {grid_resolution}° grid cells...")

    # Create bins
    lon_bins = np.arange(-180, 181, grid_resolution)
    lat_bins = np.arange(-90, 91, grid_resolution)  # Full latitude range

    # Assign each point to a bin
    df_geo["lon_bin"] = pd.cut(df_geo["long_EW"], bins=lon_bins, labels=False)
    df_geo["lat_bin"] = pd.cut(df_geo["lat_NS"], bins=lat_bins, labels=False)

    # Average values within each bin
    binned = df_geo.groupby(["lon_bin", "lat_bin"])["q"].mean().reset_index()

    # Convert bin indices back to coordinates (use bin centers)
    binned["long_EW"] = lon_bins[binned["lon_bin"].astype(int)] + grid_resolution / 2
    binned["lat_NS"] = lat_bins[binned["lat_bin"].astype(int)] + grid_resolution / 2

    print(f"  Reduced to {len(binned):,} grid cells")

    # Perform interpolation using griddata (linear is fast and stable)
    print("  Performing griddata interpolation (linear)...")
    points = binned[["long_EW", "lat_NS"]].values
    values = binned["q"].values

    geo_grid = griddata(
        points, values, (lon_grid, lat_grid), method="linear", fill_value=np.nan
    )

    # Fill NaN values using nearest neighbor for edges
    print("  Filling gaps with nearest neighbor...")
    mask = np.isnan(geo_grid)
    if mask.any():
        geo_grid_nn = griddata(points, values, (lon_grid, lat_grid), method="nearest")
        geo_grid[mask] = geo_grid_nn[mask]

    # Apply smoothing to reduce artifacts
    print("  Applying Gaussian smoothing...")
    geo_grid = gaussian_filter(geo_grid, sigma=smoothing_sigma)

    # Mask unrealistic values
    geo_grid = np.ma.masked_where((geo_grid < 0) | (geo_grid > 300), geo_grid)

    print(
        f"  Interpolated range: {np.nanmin(geo_grid):.1f} - {np.nanmax(geo_grid):.1f} mW/m²"
    )
    print("  ✓ Interpolation complete")

    # Return both data and the actual extent used
    actual_extent = (lon_min, lon_max, lat_min, lat_max)
    return geo_grid, actual_extent


def bin_data(df, grid_resolution, lon_range=(-180, 180), lat_range=(-90, 90)):
    """
    Helper function to bin geothermal data.

    Parameters:
    -----------
    df : DataFrame
        Geothermal data with columns 'long_EW', 'lat_NS', 'q'
    grid_resolution : float
        Bin size in degrees
    lon_range : tuple
        (min_lon, max_lon) for binning extent
    lat_range : tuple
        (min_lat, max_lat) for binning extent

    Returns:
    --------
    DataFrame with binned data
    """
    lon_bins = np.arange(lon_range[0], lon_range[1] + grid_resolution, grid_resolution)
    lat_bins = np.arange(lat_range[0], lat_range[1] + grid_resolution, grid_resolution)

    df = df.copy()
    df["lon_bin"] = pd.cut(df["long_EW"], bins=lon_bins, labels=False)
    df["lat_bin"] = pd.cut(df["lat_NS"], bins=lat_bins, labels=False)

    # Use mean within bin for consistency
    binned = df.groupby(["lon_bin", "lat_bin"])["q"].mean().reset_index()

    binned["long_EW"] = lon_bins[binned["lon_bin"].astype(int)] + grid_resolution / 2
    binned["lat_NS"] = lat_bins[binned["lat_bin"].astype(int)] + grid_resolution / 2

    return binned


def interpolate_geothermal_grid(
    df_geo,
    extent,
    target_shape,
    grid_resolution=2.0,
    smoothing_sigma=0.5,
    buffer_degrees=0,
):
    """
    Modular geothermal interpolation function for consistent processing.
    Uses griddata (linear + nearest neighbor) approach for consistency.

    Parameters:
    -----------
    df_geo : DataFrame
        Geothermal point data with columns 'long_EW', 'lat_NS', 'q'
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max) for output grid
    target_shape : tuple
        (height, width) of output grid
    grid_resolution : float
        Bin size in degrees (coarser = fewer gaps, finer = more detail)
    smoothing_sigma : float
        Gaussian smoothing sigma (higher = smoother)
    buffer_degrees : float
        Extend data selection by this many degrees beyond extent

    Returns:
    --------
    2D numpy array with interpolated heat flow values
    """
    from scipy.interpolate import griddata

    lon_min, lon_max, lat_min, lat_max = extent
    height, width = target_shape

    # Expand data selection if buffer requested
    if buffer_degrees > 0:
        df_expanded = df_geo[
            (df_geo["lat_NS"] >= lat_min - buffer_degrees)
            & (df_geo["lat_NS"] <= lat_max + buffer_degrees)
            & (df_geo["long_EW"] >= lon_min - buffer_degrees)
            & (df_geo["long_EW"] <= lon_max + buffer_degrees)
        ].copy()

        bin_extent_lon = (lon_min - buffer_degrees, lon_max + buffer_degrees)
        bin_extent_lat = (lat_min - buffer_degrees, lat_max + buffer_degrees)
    else:
        df_expanded = df_geo[
            (df_geo["lat_NS"] >= lat_min)
            & (df_geo["lat_NS"] <= lat_max)
            & (df_geo["long_EW"] >= lon_min)
            & (df_geo["long_EW"] <= lon_max)
        ].copy()

        bin_extent_lon = (lon_min, lon_max)
        bin_extent_lat = (lat_min, lat_max)

    print(f"  Using {len(df_expanded):,} measurements for interpolation")

    # Create output grid
    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)  # Reversed for image coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Bin the data
    print(f"  Binning data to {grid_resolution}° grid cells...")
    binned = bin_data(df_expanded, grid_resolution, bin_extent_lon, bin_extent_lat)
    print(f"  Binned to {len(binned):,} grid cells")

    # Interpolate using griddata (linear)
    print("  Performing griddata interpolation (linear)...")
    points = binned[["long_EW", "lat_NS"]].values
    values = binned["q"].values

    geo_grid = griddata(
        points, values, (lon_grid, lat_grid), method="linear", fill_value=np.nan
    )

    # Fill gaps with nearest neighbor
    mask = np.isnan(geo_grid)
    if mask.any():
        print(f"  Filling {mask.sum():,} gaps with nearest neighbor...")
        geo_grid_nn = griddata(points, values, (lon_grid, lat_grid), method="nearest")
        geo_grid[mask] = geo_grid_nn[mask]

    # Apply smoothing
    print(f"  Applying Gaussian smoothing (sigma={smoothing_sigma})...")
    geo_grid = gaussian_filter(geo_grid, sigma=smoothing_sigma)

    return geo_grid


def create_base_map(extent, solar_data_shape):
    """
    Create base map with common features (land, ocean, coastlines, etc).
    Returns fig, ax, land_mask, font_prop.
    """
    # Setup font
    font_props = setup_space_mono_font()
    font_prop = font_props.get("regular") if font_props else None

    # Create figure
    fig = plt.figure(
        figsize=MAP_STYLE["figsize_global"], facecolor=COLORS["background"]
    )
    ax = plt.axes(projection=ccrs.Robinson())

    # Add base map features
    add_base_features(ax, style="global")

    # Set global extent
    ax.set_global()

    # Create land mask
    print("\nCreating land mask...")
    lon_min, lon_max, lat_min, lat_max = extent
    height, width = solar_data_shape
    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    land_mask = create_land_mask(lon_grid, lat_grid, resolution="110m")
    print(f"  Land covers {land_mask.sum() / land_mask.size * 100:.1f}% of grid")

    return fig, ax, land_mask, font_prop


def create_energy_potential_map(
    solar_data, geo_data, solar_extent, geo_extent, save_path=None
):
    """
    Create combined map showing both solar and geothermal potential with distinct colors.
    Handles different extents for solar and geothermal data.
    """
    print("\n" + "=" * 60)
    print("Creating COMBINED energy potential map...")
    print("=" * 60)

    # Use geothermal extent since it's wider (includes polar regions)
    fig, ax, land_mask_geo, font_prop = create_base_map(geo_extent, geo_data.shape)

    # Convert geothermal heat flow to depth
    print("  Converting geothermal heat flow to drilling depth...")
    geo_depth = heat_flow_to_depth(geo_data)

    # Define thresholds for binary classification (combined map uses single threshold per resource)
    # Use middle solar threshold for "good" solar resource
    solar_threshold = SOLAR_THRESHOLDS[2]  # 4.0 kWh/kWp/day

    # Use "good" geothermal threshold (6 km depth - proven EGS technology)
    geo_depth_threshold = GEOTHERMAL_DEPTH_THRESHOLDS[1]  # 6 km
    print(
        f"  Using thresholds: Solar ≥{solar_threshold} kWh/kWp/day, Geothermal ≤{geo_depth_threshold} km depth"
    )

    # Create land mask for solar data extent
    land_mask_solar = create_base_map(solar_extent, solar_data.shape)[2]
    plt.close()  # Close the temporary figure

    # Create masks for areas meeting thresholds
    # Note: For depth, LOWER is BETTER (shallower drilling)
    high_solar = (solar_data >= solar_threshold) & land_mask_solar
    high_geo = (
        (geo_depth <= geo_depth_threshold) & land_mask_geo & np.isfinite(geo_depth)
    )

    # Define colors - pitch deck palette (elegant, clear, purposeful)
    solar_color = "#E8A33D"  # Rich amber/golden for solar (warm, sunny, refined)
    geo_color = "#C94545"  # Rich burgundy red for geothermal (earthy, warm, distinct)
    both_color = (
        "#8B5FBF"  # Rich violet purple for combined (premium, clearly distinct)
    )

    # Plot layers from background to foreground
    print("\nPlotting energy potential layers...")

    # Plot geothermal first (covers full extent including polar regions)
    if high_geo.any():
        geo_masked = np.ma.masked_where(~high_geo, np.ones_like(geo_depth))
        ax.imshow(
            geo_masked,
            extent=geo_extent,
            transform=ccrs.PlateCarree(),
            cmap=ListedColormap([geo_color]),
            alpha=0.6,
            zorder=2,
        )
        print(
            f"  ✓ Plotted geothermal potential (≤{geo_depth_threshold} km depth): {high_geo.sum():,} cells"
        )

    # Plot solar on top (limited extent -60 to 65)
    if high_solar.any():
        solar_masked = np.ma.masked_where(~high_solar, np.ones_like(solar_data))
        ax.imshow(
            solar_masked,
            extent=solar_extent,
            transform=ccrs.PlateCarree(),
            cmap=ListedColormap([solar_color]),
            alpha=0.6,
            zorder=3,
        )
        print(
            f"  ✓ Plotted solar potential (>{solar_threshold} kWh/kWp/day): {high_solar.sum():,} cells"
        )

    # Calculate normalized positions from top (lat=90)
    geo_lat_span = geo_extent[3] - geo_extent[2]  # 180 degrees

    # Row index for solar's TOP latitude (65°) - measured from geo's top (90°)
    solar_top_offset = geo_extent[3] - solar_extent[3]  # 90 - 65 = 25 degrees from top
    geo_lat_max_idx = int((solar_top_offset / geo_lat_span) * geo_data.shape[0])

    # Row index for solar's BOTTOM latitude (-60°) - measured from geo's top (90°)
    solar_bottom_offset = (
        geo_extent[3] - solar_extent[2]
    )  # 90 - (-60) = 150 degrees from top
    geo_lat_min_idx = int((solar_bottom_offset / geo_lat_span) * geo_data.shape[0])

    geo_depth_subset = geo_depth[geo_lat_max_idx:geo_lat_min_idx, :]

    # Resize to match solar data shape if needed
    if geo_depth_subset.shape != solar_data.shape:
        from scipy.ndimage import zoom

        zoom_factors = (
            solar_data.shape[0] / geo_depth_subset.shape[0],
            solar_data.shape[1] / geo_depth_subset.shape[1],
        )
        geo_depth_subset = zoom(geo_depth_subset, zoom_factors, order=1)

    high_geo_subset = (
        (geo_depth_subset <= geo_depth_threshold)
        & land_mask_solar
        & np.isfinite(geo_depth_subset)
    )
    both_energy = high_solar & high_geo_subset

    # Plot combined potential areas in purple
    if both_energy.any():
        both_masked = np.ma.masked_where(~both_energy, np.ones_like(solar_data))
        ax.imshow(
            both_masked,
            extent=solar_extent,
            transform=ccrs.PlateCarree(),
            cmap=ListedColormap([both_color]),
            alpha=0.7,
            zorder=4,
        )
        print(
            f"  ✓ Plotted combined potential (both conditions): {both_energy.sum():,} cells"
        )

    # Add subtitle
    plt.figtext(
        0.5,
        0.92,
        "Heat-flow and photovoltaic power potential indicate potential for solar and enhanced geothermal energy",
        fontsize=14,
        fontproperties=font_prop,
        ha="center",
        va="center",
        color="#444444",
    )

    # Create legend
    legend_x = 0.12
    legend_y = 0.48
    y_offset = 0.04

    # Solar potential
    create_legend_item(
        fig,
        legend_x,
        legend_y - y_offset,
        solar_color,
        "Solar PV Potential",
        f"PV output ≥ {solar_threshold} kWh/kWp/day",
        font_prop,
        alpha=0.7,
    )

    # Geothermal potential
    create_legend_item(
        fig,
        legend_x,
        legend_y - y_offset * 2.5,
        geo_color,
        "Geothermal Potential",
        f"Drilling depth ≤ {geo_depth_threshold} km (proven EGS)",
        font_prop,
        alpha=0.7,
    )

    # Combined potential
    create_legend_item(
        fig,
        legend_x,
        legend_y - y_offset * 4,
        both_color,
        "Combined Potential",
        "Both solar and geothermal available",
        font_prop,
        alpha=0.8,
    )

    # Add data note and attribution
    data_note = (
        "SOLAR DATA: GLOBAL SOLAR ATLAS 2.0 (SOLARGIS) | "
        "GEOTHERMAL DATA: IHFC GLOBAL HEAT FLOW DATABASE 2024"
    )

    plt.figtext(
        0.12,
        0.06,
        data_note,
        fontsize=9,
        color="#505050",
        ha="left",
        va="bottom",
        fontproperties=font_prop,
    )

    # Add Deep Sky icon
    add_deepsky_icon(ax)

    # Adjust layout
    plt.subplots_adjust(left=0.01, right=1.02, top=0.88, bottom=0.08)

    # Save if path provided
    save_map(fig, save_path)

    print("\n✓ Map creation complete")
    plt.close(fig)
    return fig


def create_solar_gradient_map(solar_data, extent, land_mask, save_path=None):
    """
    Create solar-only map with 3-color gradient showing intensity levels.
    """
    print("\n" + "=" * 60)
    print("Creating SOLAR gradient map...")
    print("=" * 60)

    fig, ax, _, font_prop = create_base_map(extent, solar_data.shape)

    # Use global solar thresholds and colors
    thresholds = SOLAR_THRESHOLDS
    colors = SOLAR_COLORS

    # Create masks for each level (only on land)
    levels = []
    for i in range(len(thresholds)):
        if i < len(thresholds) - 1:
            level = (
                (solar_data >= thresholds[i])
                & (solar_data < thresholds[i + 1])
                & land_mask
            )
        else:
            level = (solar_data >= thresholds[i]) & land_mask
        levels.append(level)

    # Plot layers
    print("\nPlotting solar potential gradient...")

    for i, (level, color) in enumerate(zip(levels, colors)):
        if level.any():
            masked = np.ma.masked_where(~level, np.ones_like(solar_data))
            ax.imshow(
                masked,
                extent=extent,
                transform=ccrs.PlateCarree(),
                cmap=ListedColormap([color]),
                alpha=0.7,
                zorder=2 + i,
            )
            if i < len(thresholds) - 1:
                print(
                    f"  ✓ Level {i + 1} ({thresholds[i]}-{thresholds[i + 1]} kWh/kWp/day): {level.sum():,} cells"
                )
            else:
                print(
                    f"  ✓ Level {i + 1} (≥{thresholds[i]} kWh/kWp/day): {level.sum():,} cells"
                )

    # Add subtitle
    plt.figtext(
        0.5,
        0.92,
        "Solar photovoltaic potential",
        fontsize=14,
        fontproperties=font_prop,
        ha="center",
        va="center",
        color="#444444",
    )

    # Create legend
    legend_x = 0.12
    legend_y = 0.50
    y_offset = 0.035

    # Use global solar threshold labels
    legend_labels = SOLAR_THRESHOLD_LABELS

    for i, (color, (label, desc)) in enumerate(zip(colors, legend_labels)):
        create_legend_item(
            fig,
            legend_x,
            legend_y - y_offset * (i + 1),
            color,
            label,
            desc,
            font_prop,
            alpha=0.7,
        )

    # Add data note
    plt.figtext(
        0.12,
        0.06,
        "SOLAR DATA: GLOBAL SOLAR ATLAS 2.0 (SOLARGIS)",
        fontsize=9,
        color="#505050",
        ha="left",
        va="bottom",
        fontproperties=font_prop,
    )

    # Add Deep Sky icon
    add_deepsky_icon(ax)

    plt.subplots_adjust(left=0.01, right=1.02, top=0.88, bottom=0.08)

    # Save if path provided
    save_map(fig, save_path)

    print("\n✓ Solar map creation complete")
    plt.close(fig)
    return fig


def create_geothermal_gradient_map(geo_data, geo_extent, save_path=None):
    """
    Create geothermal-only map showing depth to target temperature for EGS.
    Converts heat flow (mW/m²) to depth to reach 150°C.
    """
    print("\n" + "=" * 60)
    print("Creating GEOTHERMAL gradient map...")
    print("=" * 60)

    fig, ax, land_mask, font_prop = create_base_map(geo_extent, geo_data.shape)

    # Convert heat flow to depth-to-target temperature
    # Formula: Depth (km) = (T_target - T_surface) / Geothermal_Gradient
    # Where: Geothermal_Gradient (°C/km) = Heat_Flow (mW/m²) / Thermal_Conductivity (W/m·K)

    T_TARGET = 150  # °C - target temperature for EGS
    T_SURFACE = 15  # °C - average surface temperature
    THERMAL_CONDUCTIVITY = 2.5  # W/m·K - average continental crust

    print(f"\nConverting heat flow to depth-to-target temperature:")
    print(f"  Target temperature: {T_TARGET}°C")
    print(f"  Surface temperature: {T_SURFACE}°C")
    print(f"  Thermal conductivity: {THERMAL_CONDUCTIVITY} W/m·K")

    # Calculate geothermal gradient (°C/km)
    geothermal_gradient = geo_data / THERMAL_CONDUCTIVITY

    # Calculate depth to reach target temperature (km)
    depth_to_target = (T_TARGET - T_SURFACE) / geothermal_gradient

    # Print statistics
    valid_depths = depth_to_target[
        ~np.isnan(depth_to_target) & ~np.isinf(depth_to_target)
    ]
    if len(valid_depths) > 0:
        print(f"  Depth range: {valid_depths.min():.1f} - {valid_depths.max():.1f} km")
        print(f"  Median depth: {np.median(valid_depths):.1f} km")

    # Define thresholds based on drilling depth feasibility
    # Shallower = better/easier to access
    # Based on examples: 100 mW/m² → 3.4km, 60 mW/m² → 5.6km, 40 mW/m² → 8.4km
    # Using higher thresholds to show widespread potential
    thresholds = [4, 6, 8]  # km depth thresholds (shallower is better)

    # Create masks for each level (only on land)
    # NOTE: For depth, SHALLOWER is BETTER, so logic is reversed
    # Level 1: depth <= 4 km (EXCELLENT - high heat flow ~85+ mW/m²)
    # Level 2: 4 km < depth <= 6 km (GOOD - moderate heat flow ~56-85 mW/m²)
    # Level 3: 6 km < depth <= 8 km (MODERATE - lower heat flow ~42-56 mW/m²)
    # Level 4: depth > 8 km (CHALLENGING - low heat flow <42 mW/m²)

    levels = []
    # Excellent: <= 4 km (high heat flow regions)
    levels.append(
        (depth_to_target <= thresholds[0]) & land_mask & np.isfinite(depth_to_target)
    )
    # Good: 4-6 km (moderate heat flow)
    levels.append(
        (depth_to_target > thresholds[0])
        & (depth_to_target <= thresholds[1])
        & land_mask
        & np.isfinite(depth_to_target)
    )
    # Moderate: 6-8 km (lower but viable heat flow)
    levels.append(
        (depth_to_target > thresholds[1])
        & (depth_to_target <= thresholds[2])
        & land_mask
        & np.isfinite(depth_to_target)
    )
    # Challenging: > 8 km (very low heat flow)
    levels.append(
        (depth_to_target > thresholds[2]) & land_mask & np.isfinite(depth_to_target)
    )

    # Color scheme: DARKEST for shallowest/best (reversed from heat flow)
    # Dark red = shallow/excellent, Light red = deep/challenging
    colors = [
        "#8B2323",  # Dark burgundy/maroon - EXCELLENT (≤3 km)
        "#C94545",  # Medium burgundy red - GOOD (3-4 km)
        "#DE8A80",  # Light coral/pink - MODERATE (4-6 km)
        "#F2C4BE",  # Very light coral - CHALLENGING (>6 km)
    ]

    # Plot layers
    print("\nPlotting depth-to-target temperature...")

    depth_labels = ["≤4 km", "4-6 km", "6-8 km", ">8 km"]
    quality_labels = ["Excellent", "Good", "Moderate", "Challenging"]

    for i, (level, color, depth_label, quality) in enumerate(
        zip(levels, colors, depth_labels, quality_labels)
    ):
        if level.any():
            masked = np.ma.masked_where(~level, np.ones_like(depth_to_target))
            ax.imshow(
                masked,
                extent=geo_extent,
                transform=ccrs.PlateCarree(),
                cmap=ListedColormap([color]),
                alpha=0.7,
                zorder=2 + i,
            )
            print(
                f"  ✓ Level {i + 1} ({depth_label}, {quality}): {level.sum():,} cells"
            )

    # Add subtitle
    plt.figtext(
        0.5,
        0.92,
        f"Drilling depth to reach {T_TARGET}°C for enhanced geothermal systems",
        fontsize=14,
        fontproperties=font_prop,
        ha="center",
        va="center",
        color="#444444",
    )

    # Create legend
    legend_x = 0.12
    legend_y = 0.50
    y_offset = 0.038

    legend_labels = GEOTHERMAL_DEPTH_LABELS

    for i, (color, (label, desc)) in enumerate(zip(colors, legend_labels)):
        create_legend_item(
            fig,
            legend_x,
            legend_y - y_offset * (i + 1),
            color,
            label,
            desc,
            font_prop,
            alpha=0.7,
        )

    # Add data note
    plt.figtext(
        0.12,
        0.06,
        "GEOTHERMAL DATA: IHFC GLOBAL HEAT FLOW DATABASE 2024",
        fontsize=9,
        color="#505050",
        ha="left",
        va="bottom",
        fontproperties=font_prop,
    )

    # Add Deep Sky icon
    add_deepsky_icon(ax)

    plt.subplots_adjust(left=0.01, right=1.02, top=0.88, bottom=0.08)

    # Save if path provided
    save_map(fig, save_path)

    print("\n✓ Geothermal map creation complete")
    plt.close(fig)
    return fig


def create_regional_geothermal_map(
    df_geo,
    region_name,
    lat_range,
    lon_range,
    save_path=None,
    country_shapefile=None,
    country_geom=None,
    plot_lat_range=None,
    plot_lon_range=None,
):
    """
    Create a detailed regional geothermal map using all available data points.

    Parameters:
    -----------
    df_geo : DataFrame
        Full geothermal point data
    region_name : str
        Name of the region for title
    lat_range : tuple
        (min_lat, max_lat) for the region
    lon_range : tuple
        (min_lon, max_lon) for the region
    save_path : str, optional
        Path to save the figure
    country_shapefile : str, optional
        Path to country shapefile (without extension) for precise masking.
        If provided, data will only be shown within country borders.
    country_geom : shapely geometry, optional
        Pre-loaded country/region geometry. Takes precedence over country_shapefile.
    """
    print("\n" + "=" * 60)
    print(f"Creating REGIONAL geothermal map: {region_name}")
    print("=" * 60)

    # Load country shapefile if provided and no geometry passed
    if country_geom is None and country_shapefile:
        country_geom, lat_range, lon_range = load_country_shapefile(country_shapefile)

    # Use plot bounds if provided, otherwise use data bounds
    if plot_lat_range is None:
        plot_lat_range = lat_range
    if plot_lon_range is None:
        plot_lon_range = lon_range

    # Expand the data selection region to improve edge interpolation
    # Pull data from 5 degrees beyond boundaries
    buffer_degrees = 0

    df_region_expanded = df_geo[
        (df_geo["lat_NS"] >= lat_range[0] - buffer_degrees)
        & (df_geo["lat_NS"] <= lat_range[1] + buffer_degrees)
        & (df_geo["long_EW"] >= lon_range[0] - buffer_degrees)
        & (df_geo["long_EW"] <= lon_range[1] + buffer_degrees)
    ].copy()

    # Also keep track of actual region data for statistics and plotting
    df_region = df_geo[
        (df_geo["lat_NS"] >= lat_range[0])
        & (df_geo["lat_NS"] <= lat_range[1])
        & (df_geo["long_EW"] >= lon_range[0])
        & (df_geo["long_EW"] <= lon_range[1])
    ].copy()

    print(f"  Found {len(df_region):,} measurements in region")
    print(
        f"  Using {len(df_region_expanded):,} measurements (including {buffer_degrees}° buffer)"
    )

    if len(df_region_expanded) == 0:
        print(f"  WARNING: No data points found in {region_name}")
        return None

    print(
        f"  Heat flow range: {df_region['q'].min():.1f} - {df_region['q'].max():.1f} mW/m²"
    )

    # Calculate figure dimensions based on region aspect ratio
    lat_span = lat_range[1] - lat_range[0]
    lon_span = lon_range[1] - lon_range[0]

    # Use Plate Carree for regional maps (better for smaller areas)
    # Aspect ratio accounting for latitude distortion
    lat_center = (lat_range[0] + lat_range[1]) / 2
    aspect_correction = np.cos(np.radians(lat_center))
    aspect_ratio = (lon_span * aspect_correction) / lat_span

    # Set figure width and calculate height
    fig_width = 16
    fig_height = fig_width / aspect_ratio
    fig_height = np.clip(fig_height, 8, 12)  # Keep reasonable bounds

    # Setup font
    font_props = setup_space_mono_font()
    font_prop = font_props.get("regular") if font_props else None

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=COLORS["background"])
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent (use plot bounds which may include buffer)
    ax.set_extent(
        [plot_lon_range[0], plot_lon_range[1], plot_lat_range[0], plot_lat_range[1]],
        crs=ccrs.PlateCarree(),
    )

    # Add map features
    add_base_features(ax, style="regional")

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="#AAAAAA", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # Use modular interpolation function for consistency with global maps
    # Regional maps can have finer output resolution while using same interpolation approach
    target_shape = (
        int(lat_span / 0.25),
        int(lon_span / 0.25),
    )  # 0.25 degree output grid

    print(f"  Interpolating to {target_shape} grid...")

    # Use same interpolation parameters as global map for consistency
    # Finer grid_resolution for regional detail: 0.5° instead of 2°
    extent_region = (lon_range[0], lon_range[1], lat_range[0], lat_range[1])
    geo_grid = interpolate_geothermal_grid(
        df_geo,
        extent_region,
        target_shape,
        grid_resolution=0.25,  # Finer than global (2°) for regional detail
        smoothing_sigma=0.5,  # Same as global
        buffer_degrees=buffer_degrees,  # Use buffer for better edge interpolation
    )

    # Create coordinate grids for masking
    lons = np.linspace(lon_range[0], lon_range[1], target_shape[1])
    lats = np.linspace(lat_range[1], lat_range[0], target_shape[0])  # Reversed
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Calculate depth to target temperature
    T_TARGET = 150
    T_SURFACE = 15
    THERMAL_CONDUCTIVITY = 2.5

    geothermal_gradient = geo_grid / THERMAL_CONDUCTIVITY
    depth_to_target = (T_TARGET - T_SURFACE) / geothermal_gradient

    # Create land mask for this region
    print("  Creating land mask...")
    land_mask = create_land_mask(
        lon_grid, lat_grid, country_geom=country_geom, resolution="50m"
    )
    mask_type = "Country" if country_geom else "Land"
    print(
        f"  {mask_type} mask covers {land_mask.sum() / land_mask.size * 100:.1f}% of grid"
    )

    # Define depth thresholds (same as global)
    thresholds = [4, 6, 8]

    levels = []
    levels.append(
        (depth_to_target <= thresholds[0]) & land_mask & np.isfinite(depth_to_target)
    )
    levels.append(
        (depth_to_target > thresholds[0])
        & (depth_to_target <= thresholds[1])
        & land_mask
        & np.isfinite(depth_to_target)
    )
    levels.append(
        (depth_to_target > thresholds[1])
        & (depth_to_target <= thresholds[2])
        & land_mask
        & np.isfinite(depth_to_target)
    )
    levels.append(
        (depth_to_target > thresholds[2]) & land_mask & np.isfinite(depth_to_target)
    )

    colors = [
        "#8B2323",  # Dark burgundy - Excellent
        "#C94545",  # Medium burgundy - Good
        "#DE8A80",  # Light coral - Moderate
        "#F2C4BE",  # Very light coral - Challenging
    ]

    # Plot layers
    extent_plot = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]

    depth_labels = ["≤4 km", "4-6 km", "6-8 km", ">8 km"]
    quality_labels = ["Excellent", "Good", "Moderate", "Challenging"]

    print("  Plotting depth-to-target layers...")
    for i, (level, color, depth_label, quality) in enumerate(
        zip(levels, colors, depth_labels, quality_labels)
    ):
        if level.any():
            masked = np.ma.masked_where(~level, np.ones_like(depth_to_target))
            ax.imshow(
                masked,
                extent=extent_plot,
                transform=ccrs.PlateCarree(),
                cmap=ListedColormap([color]),
                alpha=0.8,
                zorder=2 + i,
                interpolation="bilinear",
            )
            print(f"    ✓ {depth_label} ({quality}): {level.sum():,} cells")

    # Note: Data point filtering commented out - not displayed on regional maps
    # If needed in future, can use country_geom with shapely Point.contains() logic
    print(f"  Total measurements in region: {len(df_region):,}")

    # Plot actual data points as small dots (only on land)
    # ax.scatter(df_region_land['long_EW'], df_region_land['lat_NS'],
    #           c='black', s=0.5, alpha=0.3, zorder=10,
    #           transform=ccrs.PlateCarree(),
    #           label=f'{len(df_region_land)} measurements')

    # # Add title
    # plt.figtext(0.5, 0.95,
    #            f'{region_name}: Drilling depth to reach {T_TARGET}°C',
    #            fontsize=16, fontproperties=title_font,
    #            ha='center', va='top')

    # # Create compact legend
    # legend_x = 0.08
    # legend_y = 0.30
    # square_size = 0.012
    # y_offset = 0.04

    # for i, (color, depth_label, quality) in enumerate(zip(colors, depth_labels, quality_labels)):
    #     rect = Rectangle((legend_x, legend_y - y_offset*i - 0.005), square_size, 0.010,
    #                     transform=fig.transFigure,
    #                     facecolor=color,
    #                     edgecolor='none',
    #                     alpha=0.8)
    #     fig.patches.append(rect)
    #     plt.figtext(legend_x + square_size + 0.006, legend_y - y_offset*i,
    #                f'{depth_label} - {quality}',
    #                fontsize=9, fontproperties=font_prop,
    #                ha='left', va='center')

    # Add data note
    plt.figtext(
        0.5,
        0.02,
        f"GEOTHERMAL DATA: IHFC 2024 | {len(df_region)} measurements in region",
        fontsize=12,
        color="#505050",
        ha="center",
        va="bottom",
        fontproperties=font_prop,
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Save
    save_map(fig, save_path)

    print(f"\n✓ Regional geothermal map complete: {region_name}")
    plt.close(fig)
    return fig


def create_regional_solar_map(
    solar_data_global,
    solar_extent_global,
    region_name,
    lat_range,
    lon_range,
    save_path=None,
    country_shapefile=None,
    country_geom=None,
    plot_lat_range=None,
    plot_lon_range=None,
):
    """
    Create regional solar PV potential map by extracting from global solar data.

    Parameters:
    -----------
    solar_data_global : ndarray
        Global solar PV data array
    solar_extent_global : tuple
        (lon_min, lon_max, lat_min, lat_max) for global solar data
    region_name : str
        Name of region for title
    lat_range : tuple
        (min_lat, max_lat) for the region
    lon_range : tuple
        (min_lon, max_lon) for the region
    save_path : str, optional
        Path to save the figure
    country_shapefile : str, optional
        Path to country shapefile for precise masking
    country_geom : shapely geometry, optional
        Pre-loaded country/region geometry. Takes precedence over country_shapefile.
    """
    print("\n" + "=" * 60)
    print(f"Creating REGIONAL solar map: {region_name}")
    print("=" * 60)

    # Load country shapefile if provided and no geometry passed
    if country_geom is None and country_shapefile:
        country_geom, lat_range, lon_range = load_country_shapefile(country_shapefile)

    # Use plot bounds if provided, otherwise use data bounds
    if plot_lat_range is None:
        plot_lat_range = lat_range
    if plot_lon_range is None:
        plot_lon_range = lon_range

    # Extract regional subset from global solar data
    glon_min, glon_max, glat_min, glat_max = solar_extent_global
    gheight, gwidth = solar_data_global.shape

    print(
        f"  Global solar extent: lon({glon_min}, {glon_max}), lat({glat_min}, {glat_max})"
    )
    print(
        f"  Region extent: lon({lon_range[0]}, {lon_range[1]}), lat({lat_range[0]}, {lat_range[1]})"
    )

    # Check if region overlaps with solar data extent
    if (
        lon_range[0] > glon_max
        or lon_range[1] < glon_min
        or lat_range[0] > glat_max
        or lat_range[1] < glat_min
    ):
        print(f"  WARNING: Region does not overlap with solar data extent!")
        print(f"  No solar data available for this region.")
        return None

    # Clip region to solar data extent
    clipped_lon_min = max(lon_range[0], glon_min)
    clipped_lon_max = min(lon_range[1], glon_max)
    clipped_lat_min = max(lat_range[0], glat_min)
    clipped_lat_max = min(lat_range[1], glat_max)

    print(
        f"  Clipped to solar extent: lon({clipped_lon_min:.2f}, {clipped_lon_max:.2f}), lat({clipped_lat_min:.2f}, {clipped_lat_max:.2f})"
    )

    # Calculate pixel indices for the clipped region
    lon_idx_min = int((clipped_lon_min - glon_min) / (glon_max - glon_min) * gwidth)
    lon_idx_max = int((clipped_lon_max - glon_min) / (glon_max - glon_min) * gwidth)
    lat_idx_min = int((glat_max - clipped_lat_max) / (glat_max - glat_min) * gheight)
    lat_idx_max = int((glat_max - clipped_lat_min) / (glat_max - glat_min) * gheight)

    # Clamp indices to valid range
    lon_idx_min = max(0, min(lon_idx_min, gwidth - 1))
    lon_idx_max = max(1, min(lon_idx_max, gwidth))
    lat_idx_min = max(0, min(lat_idx_min, gheight - 1))
    lat_idx_max = max(1, min(lat_idx_max, gheight))

    # Extract subset
    solar_data_region = solar_data_global[
        lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max
    ]
    print(f"  Extracted solar data shape: {solar_data_region.shape}")

    # Use clipped extent for the actual solar data
    actual_solar_extent = (
        clipped_lon_min,
        clipped_lon_max,
        clipped_lat_min,
        clipped_lat_max,
    )

    # Setup figure
    lat_span = lat_range[1] - lat_range[0]
    lon_span = lon_range[1] - lon_range[0]
    lat_center = (lat_range[0] + lat_range[1]) / 2
    aspect_correction = np.cos(np.radians(lat_center))
    aspect_ratio = (lon_span * aspect_correction) / lat_span

    fig_width = 16
    fig_height = np.clip(fig_width / aspect_ratio, 8, 12)

    font_props = setup_space_mono_font()
    font_prop = font_props.get("regular") if font_props else None

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=COLORS["background"])
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Set extent (use plot bounds which may include buffer)
    ax.set_extent(
        [plot_lon_range[0], plot_lon_range[1], plot_lat_range[0], plot_lat_range[1]],
        crs=ccrs.PlateCarree(),
    )

    # Add map features
    add_base_features(ax, style="regional")

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="#AAAAAA", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # Create land/country mask based on the actual solar data extent (clipped)
    print("  Creating land mask...")
    lons = np.linspace(clipped_lon_min, clipped_lon_max, solar_data_region.shape[1])
    lats = np.linspace(clipped_lat_max, clipped_lat_min, solar_data_region.shape[0])
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    land_mask = create_land_mask(
        lon_grid, lat_grid, country_geom=country_geom, resolution="50m"
    )

    # Use global solar thresholds and colors for consistency
    thresholds = SOLAR_THRESHOLDS
    colors = SOLAR_COLORS

    # Plot layers using the clipped extent
    extent_plot = [clipped_lon_min, clipped_lon_max, clipped_lat_min, clipped_lat_max]
    print(f"  Plotting with extent: {extent_plot}")

    for i in range(len(thresholds)):
        if i < len(thresholds) - 1:
            level = (
                (solar_data_region >= thresholds[i])
                & (solar_data_region < thresholds[i + 1])
                & land_mask
            )
        else:
            level = (solar_data_region >= thresholds[i]) & land_mask

        if level.any():
            masked = np.ma.masked_where(~level, np.ones_like(solar_data_region))
            ax.imshow(
                masked,
                extent=extent_plot,
                transform=ccrs.PlateCarree(),
                cmap=ListedColormap([colors[i]]),
                alpha=0.7,
                zorder=2 + i,
            )
            print(f"    ✓ Level {i + 1}: {level.sum():,} cells")

    # Add data note with coverage info
    coverage_note = "SOLAR DATA: GLOBAL SOLAR ATLAS 2.0 (SOLARGIS)"
    # Add note if region extends beyond solar data coverage
    if lat_range[1] > glat_max or lat_range[0] < glat_min:
        coverage_note += f" | Coverage limited to {glat_min}° to {glat_max}° latitude"

    plt.figtext(
        0.5,
        0.02,
        coverage_note,
        fontsize=12,
        color="#505050",
        ha="center",
        va="bottom",
        fontproperties=font_prop,
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Save if path provided
    save_map(fig, save_path)

    print(f"\n✓ Regional solar map complete: {region_name}")
    plt.close(fig)
    return fig


def calculate_solar_potential_twh_year(solar_data, land_mask, assumptions):
    """
    Calculate annual solar energy potential in TWh/year for a region.

    Formula:
    TWh/year = (Suitable_Area_km² × Land_Availability × Power_Density_MW/km²
                × Capacity_Factor × System_Efficiency × 8760 hours/year) / 1,000,000

    Parameters:
    -----------
    solar_data : ndarray
        Solar PV data (kWh/kWp/day)
    land_mask : ndarray
        Boolean mask indicating suitable land
    assumptions : dict
        Dictionary containing solar parameters

    Returns:
    --------
    dict: Contains total TWh/year and suitable area in km²
    """
    # Calculate suitable area (land with solar potential above minimum threshold)
    # Use the lowest threshold as minimum viable solar resource
    min_solar_threshold = assumptions["min_solar_threshold"]
    suitable_mask = (
        (solar_data >= min_solar_threshold) & land_mask & np.isfinite(solar_data)
    )

    # Count suitable cells and convert to area
    # Assuming 0.5° resolution ≈ 55km at equator
    # Area per cell varies with latitude, but for regional estimates we'll use average
    # At lat=45°, 0.5° ≈ 39km, giving ~1500 km² per cell
    # For more accuracy, we should calculate area per cell based on latitude
    # For now, use approximate area calculation

    # Calculate area per cell based on actual grid resolution
    # This is an approximation - proper calculation would account for latitude variation
    suitable_cells = np.sum(suitable_mask)

    # Rough estimate: 0.5° × 0.5° cell at mid-latitudes ≈ 3000 km²
    # Better approach: calculate based on actual extent and shape
    # For now, use conservative estimate
    km_per_degree_lon = 111.0  # at equator
    km_per_degree_lat = 111.0
    cell_area_km2 = (km_per_degree_lon * 0.5) * (km_per_degree_lat * 0.5)  # Approximate
    suitable_area_km2 = suitable_cells * cell_area_km2

    base_land_avail = assumptions["solar_land_availability"]

    if suitable_cells > 0:
        # Calculate average PVOUT in suitable areas as solar quality proxy
        avg_pvout = np.nanmean(solar_data[suitable_mask])
    else:
        avg_pvout = 0

    # Apply effective land availability factor
    available_area_km2 = suitable_area_km2 * base_land_avail * avg_pvout

    # Calculate annual energy potential
    power_density = assumptions["solar_power_density_MW_km2"]
    capacity_factor = assumptions["solar_capacity_factor"]
    efficiency = assumptions["solar_system_efficiency"]

    installed_capacity_MW = available_area_km2 * power_density
    annual_generation_MWh = installed_capacity_MW * capacity_factor * efficiency * 8760
    annual_generation_TWh = annual_generation_MWh / 1_000_000

    return {
        "TWh_per_year": annual_generation_TWh,
        "suitable_area_km2": suitable_area_km2,
        "available_area_km2": available_area_km2,
        "installed_capacity_MW": installed_capacity_MW,
    }


def calculate_geothermal_potential_mw(geo_depth_data, land_mask, assumptions):
    """
    Calculate enhanced geothermal energy potential in MW for a region.

    Formula (updated with Franzmann land eligibility):
    MW = (Viable_Area_km² × Land_Eligibility / Well_Spacing_km²) × Well_Power_MW

    Parameters:
    -----------
    geo_depth_data : ndarray
        Depth to target temperature (km) calculated from heat flow
    land_mask : ndarray
        Boolean mask indicating land areas
    assumptions : dict
        Dictionary containing geothermal parameters

    Returns:
    --------
    dict: Contains installed capacity (MW), viable area (km²), eligible area (km²), and number of wells
    """
    # Calculate viable area (land with drilling depth below max depth threshold)
    max_depth = assumptions["geo_max_depth_km"]
    viable_mask = (
        (geo_depth_data <= max_depth) & land_mask & np.isfinite(geo_depth_data)
    )

    # Count viable cells and convert to area
    viable_cells = np.sum(viable_mask)

    # Calculate area (same approximation as solar)
    km_per_degree = 111.0
    cell_area_km2 = (km_per_degree * 0.5) ** 2
    viable_area_km2 = viable_cells * cell_area_km2

    # Apply land eligibility factor (Franzmann methodology)
    # Not all land with good heat flow is available for EGS development
    # Accounts for: water scarcity, slope, settlements, protected areas, infrastructure, etc.
    land_eligibility = assumptions.get(
        "geo_land_eligibility", 0.25
    )  # Default to 25% (global avg per Franzmann)
    eligible_area_km2 = viable_area_km2 * land_eligibility

    # Calculate number of wells and power potential
    well_spacing = assumptions["geo_well_spacing_km2"]
    well_power = assumptions["geo_well_power_MW"]
    capacity_factor = assumptions["geo_capacity_factor"]

    num_wells = eligible_area_km2 / well_spacing
    installed_capacity_MW = num_wells * well_power

    return {
        "installed_capacity_MW": installed_capacity_MW,
        "viable_area_km2": viable_area_km2,
        "eligible_area_km2": eligible_area_km2,
        "num_wells": num_wells,
        "land_eligibility_factor": land_eligibility,
    }


def estimate_regional_energy_potentials(
    solar_data_global,
    solar_extent_global,
    geo_data_global,
    geo_extent_global,
    df_geo,
    save_path="data/energy/regional_estimates.csv",
):
    """
    Calculate energy potential estimates for all defined regions and save to CSV.

    Solar potential is calculated as TWh/year (annual generation).
    Geothermal potential is calculated as MW (installed capacity).

    Parameters:
    -----------
    solar_data_global : ndarray
        Global solar PV data
    solar_extent_global : tuple
        (lon_min, lon_max, lat_min, lat_max) for solar data
    geo_data_global : ndarray
        Global geothermal heat flow data (will be converted to depth)
    geo_extent_global : tuple
        (lon_min, lon_max, lat_min, lat_max) for geothermal data
    df_geo : DataFrame
        Geothermal point data for regional interpolation
    save_path : str
        Path to save CSV output

    Returns:
    --------
    DataFrame: Regional estimates with columns:
        - solar_TWh_year_{scenario}: Annual solar generation potential
        - geo_MW_{scenario}: Installed geothermal capacity potential
        - solar_area_km2_{scenario}: Suitable solar area
        - geo_area_km2_{scenario}: Viable geothermal area
    """
    print("\n" + "=" * 70)
    print("CALCULATING REGIONAL ENERGY POTENTIALS")
    print("=" * 70)

    results = []

    for region_name, region_config in REGIONS.items():
        print(f"\nProcessing {region_name}...")

        # Load country/continent geometry
        geojson_path = os.path.join(script_dir, WORLD_COUNTRIES_GEOJSON)
        try:
            country_geom, lat_range, lon_range, _, _ = load_countries_from_geojson(
                geojson_path,
                countries=region_config.get("countries"),
                continent=region_config.get("continent"),
                plot_buffer_degrees=region_config.get("plot_buffer_degrees", 0),
            )
        except Exception as e:
            print(f"  ERROR loading geometry for {region_name}: {e}")
            print(f"  Skipping this region")
            continue

        # Extract regional solar data
        glon_min, glon_max, glat_min, glat_max = solar_extent_global
        gheight, gwidth = solar_data_global.shape

        # Check overlap with solar extent
        if (
            lon_range[0] <= glon_max
            and lon_range[1] >= glon_min
            and lat_range[0] <= glat_max
            and lat_range[1] >= glat_min
        ):
            # Extract solar subset
            clipped_lon_min = max(lon_range[0], glon_min)
            clipped_lon_max = min(lon_range[1], glon_max)
            clipped_lat_min = max(lat_range[0], glat_min)
            clipped_lat_max = min(lat_range[1], glat_max)

            lon_idx_min = int(
                (clipped_lon_min - glon_min) / (glon_max - glon_min) * gwidth
            )
            lon_idx_max = int(
                (clipped_lon_max - glon_min) / (glon_max - glon_min) * gwidth
            )
            lat_idx_min = int(
                (glat_max - clipped_lat_max) / (glat_max - glat_min) * gheight
            )
            lat_idx_max = int(
                (glat_max - clipped_lat_min) / (glat_max - glat_min) * gheight
            )

            lon_idx_min = max(0, min(lon_idx_min, gwidth - 1))
            lon_idx_max = max(1, min(lon_idx_max, gwidth))
            lat_idx_min = max(0, min(lat_idx_min, gheight - 1))
            lat_idx_max = max(1, min(lat_idx_max, gheight))

            solar_data_region = solar_data_global[
                lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max
            ]

            # Create land mask for solar region using country boundaries
            lons_solar = np.linspace(
                clipped_lon_min, clipped_lon_max, solar_data_region.shape[1]
            )
            lats_solar = np.linspace(
                clipped_lat_max, clipped_lat_min, solar_data_region.shape[0]
            )
            lon_grid_solar, lat_grid_solar = np.meshgrid(lons_solar, lats_solar)
            land_mask_solar = create_land_mask(
                lon_grid_solar,
                lat_grid_solar,
                country_geom=country_geom,
                resolution="110m",
            )
        else:
            solar_data_region = None
            land_mask_solar = None

        # Extract/interpolate regional geothermal data
        # Use regional interpolation for better accuracy
        target_shape = (
            int((lat_range[1] - lat_range[0]) / 0.5),
            int((lon_range[1] - lon_range[0]) / 0.5),
        )
        extent_region = (lon_range[0], lon_range[1], lat_range[0], lat_range[1])

        geo_grid_region = interpolate_geothermal_grid(
            df_geo,
            extent_region,
            target_shape,
            grid_resolution=1.0,
            smoothing_sigma=0.5,
            buffer_degrees=5,
        )

        # Convert heat flow to depth
        geo_depth_region = heat_flow_to_depth(geo_grid_region)

        # Create land mask for geothermal region using country boundaries
        lons_geo = np.linspace(lon_range[0], lon_range[1], target_shape[1])
        lats_geo = np.linspace(lat_range[1], lat_range[0], target_shape[0])
        lon_grid_geo, lat_grid_geo = np.meshgrid(lons_geo, lats_geo)
        land_mask_geo = create_land_mask(
            lon_grid_geo, lat_grid_geo, country_geom=country_geom, resolution="110m"
        )

        # Calculate estimates for each scenario
        row = {"region": region_name}

        for scenario in ["conservative", "moderate", "optimistic", "miyake_equivalent"]:
            assumptions = ENERGY_ASSUMPTIONS[scenario]

            # Solar potential
            if solar_data_region is not None and land_mask_solar is not None:
                solar_result = calculate_solar_potential_twh_year(
                    solar_data_region, land_mask_solar, assumptions
                )
                row[f"solar_TWh_year_{scenario}"] = solar_result["TWh_per_year"]
                row[f"solar_area_km2_{scenario}"] = solar_result["suitable_area_km2"]
            else:
                row[f"solar_TWh_year_{scenario}"] = 0
                row[f"solar_area_km2_{scenario}"] = 0

            # Geothermal potential
            geo_result = calculate_geothermal_potential_mw(
                geo_depth_region, land_mask_geo, assumptions
            )
            row[f"geo_MW_{scenario}"] = geo_result["installed_capacity_MW"]
            row[f"geo_area_km2_{scenario}"] = geo_result["viable_area_km2"]

        results.append(row)

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Reorder columns for clarity
    column_order = ["region"]
    for scenario in ["conservative", "moderate", "optimistic", "miyake_equivalent"]:
        column_order.extend(
            [
                f"solar_TWh_year_{scenario}",
                f"geo_MW_{scenario}",
                f"solar_area_km2_{scenario}",
                f"geo_area_km2_{scenario}",
            ]
        )

    df_results = df_results[column_order]
    print(df_results[["region", "solar_TWh_year_miyake_equivalent"]])

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_results.to_csv(save_path, index=False, float_format="%.2f")

    print(f"\n✓ Regional estimates saved to: {save_path}")

    return df_results


def main():
    """
    Main function to generate all three energy potential maps for CDR operations.
    """
    print("\n" + "=" * 70)
    print("ENERGY POTENTIAL VISUALIZATION FOR CDR OPERATIONS")
    print("Generating 3 maps: Combined, Solar, and Geothermal")
    print("=" * 70)

    # Load solar data
    solar_data, solar_extent = load_solar_data()

    # Load geothermal data
    df_geo = load_geothermal_data()

    # Interpolate geothermal data with FULL latitude coverage for global view
    geo_data, geo_extent = interpolate_geothermal(
        df_geo, solar_extent, solar_data.shape, use_full_latitude=True
    )

    # Create land mask for solar maps
    print("\n" + "=" * 60)
    print("Creating land mask for solar extent...")
    print("=" * 60)
    _, _, land_mask_solar, _ = create_base_map(solar_extent, solar_data.shape)
    plt.close()  # Close the base map figure

    # Generate all three maps
    print("\n" + "=" * 70)
    print("GENERATING MAPS")
    print("=" * 70)

    # 1. Combined map (uses extended geothermal extent)
    combined_path = "figures/cdr_energy_potential.png"
    create_energy_potential_map(
        solar_data, geo_data, solar_extent, geo_extent, save_path=combined_path
    )

    # 2. Solar gradient map (limited to solar extent)
    solar_path = "figures/cdr_solar_map.png"
    create_solar_gradient_map(
        solar_data, solar_extent, land_mask_solar, save_path=solar_path
    )

    # 3. Geothermal gradient map (full global extent)
    geo_path = "figures/cdr_geothermal_map.png"
    create_geothermal_gradient_map(geo_data, geo_extent, save_path=geo_path)

    # 4. Calculate regional energy potentials
    print("\n" + "=" * 70)
    print("CALCULATING REGIONAL ENERGY POTENTIALS")
    print("=" * 70)

    # Generate regional estimates CSV
    df_estimates = estimate_regional_energy_potentials(
        solar_data,
        solar_extent,
        geo_data,
        geo_extent,
        df_geo,
        save_path="data/energy/regional_estimates.csv",
    )

    # 5. Regional maps (geothermal and solar)
    print("\n" + "=" * 70)
    print("GENERATING REGIONAL MAPS (GEOTHERMAL & SOLAR)")
    print("=" * 70)

    regional_outputs = {}
    for region_name, region_config in REGIONS.items():
        region_filename = sanitize_region_name(region_name)

        # Load country/continent geometry
        geojson_path = os.path.join(script_dir, WORLD_COUNTRIES_GEOJSON)
        try:
            country_geom, lat_range, lon_range, plot_lat_range, plot_lon_range = (
                load_countries_from_geojson(
                    geojson_path,
                    countries=region_config.get("countries"),
                    continent=region_config.get("continent"),
                    plot_buffer_degrees=region_config.get("plot_buffer_degrees", 0),
                )
            )
            # Create a temp shapefile path for compatibility (will load from country_geom directly)
            country_shapefile = None
        except Exception as e:
            print(f"  ERROR: Could not load geometry for {region_name}: {e}")
            print(f"  Skipping this region")
            continue

        # Geothermal map
        geo_save_path = f"figures/regional/cdr_geothermal_{region_filename}.png"
        fig_geo = create_regional_geothermal_map(
            df_geo,
            region_name,
            lat_range,
            lon_range,
            save_path=geo_save_path,
            country_shapefile=country_shapefile,
            country_geom=country_geom,
            plot_lat_range=plot_lat_range,
            plot_lon_range=plot_lon_range,
        )

        # Solar map
        solar_save_path = f"figures/regional/cdr_solar_{region_filename}.png"
        fig_solar = create_regional_solar_map(
            solar_data,
            solar_extent,
            region_name,
            lat_range,
            lon_range,
            save_path=solar_save_path,
            country_shapefile=country_shapefile,
            country_geom=country_geom,
            plot_lat_range=plot_lat_range,
            plot_lon_range=plot_lon_range,
        )

        if fig_geo is not None and fig_solar is not None:
            regional_outputs[region_name] = {
                "geothermal": geo_save_path,
                "solar": solar_save_path,
            }

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nGlobal Outputs:")
    print(f"  1. Combined:   {combined_path}")
    print(f"  2. Solar:      {solar_path}")
    print(f"  3. Geothermal: {geo_path}")
    print(
        f"\nRegional Outputs ({len(regional_outputs)} regions x 2 maps = {len(regional_outputs) * 2} total):"
    )
    for i, (region, paths) in enumerate(regional_outputs.items(), 1):
        print(f"  {region}:")
        print(f"    - Geothermal: {paths['geothermal']}")
        print(f"    - Solar:      {paths['solar']}")
    print("\nKey insights:")
    print("  - Solar potential excellent in sun belt regions")
    print("  - Geothermal potential high in tectonically active zones")
    print("  - Regional maps use same interpolation as global for consistency")
    print("  - Country-specific masking available (e.g., Canada shapefile)")
    print("\nLimitations:")
    print("  - Geothermal interpolation based on point measurements")
    print("  - Does not account for infrastructure or accessibility")
    print("  - EGS technology still under development in many regions")
    print("  - Local site assessment required for any deployment")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
