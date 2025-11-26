"""
Folium map creation and layer rendering

IMPORTANT: This module uses Web Mercator projection (EPSG:3857) for the base map,
which is the standard for web mapping tiles. All raster data must be reprojected
to Web Mercator before display. See processing/rasterize.py for details.
"""

import base64
import io
from typing import Any, Dict, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def create_base_map(
    center: list, zoom: int, tile_layer: str = "CartoDB positron"
) -> folium.Map:
    """Create base Folium map with default Web Mercator projection (EPSG:3857)."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,  # We'll add tiles manually to enable noWrap
        attr="Deep Sky Research",
        max_bounds=True,  # Constrain map to one world view
        # Uses default Web Mercator (EPSG:3857) projection for proper tile alignment
    )

    # Add tile layer with noWrap to prevent infinite horizontal scrolling
    folium.TileLayer(
        tiles=tile_layer,
        attr="Deep Sky Research",
        no_wrap=True,  # Prevent tile wrapping - keeps map to single world view
    ).add_to(m)

    return m


def add_solar_layer(
    m: folium.Map,
    data: np.ndarray,
    extent: tuple,
    name: str,
    opacity: float = 0.6,
) -> folium.Map:
    """
    Add solar PV potential layer as colored raster overlay.

    Parameters:
    -----------
    m : folium.Map
        Map to add layer to
    data : ndarray
        Solar PV potential data (kWh/kWp/day)
    extent : tuple
        (lon_min, lat_min, lon_max, lat_max)
    name : str
        Layer name
    opacity : float
        Layer opacity (0-1)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Create a colormap for solar potential (low=blue, high=red)
    colors = [
        "#313695",
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#e0f3f8",
        "#ffffbf",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#a50026",
    ]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("solar", colors, N=n_bins)

    # Normalize data to 0-1 range (typical solar values 2-7 kWh/kWp/day)
    data_clean = data.copy()
    data_clean[np.isnan(data_clean)] = 0
    vmin, vmax = 2.0, 7.0
    data_norm = np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)

    # Apply colormap
    rgba = cmap(data_norm)
    rgba = (rgba * 255).astype(np.uint8)

    # Set opacity and make no-data transparent
    rgba[:, :, 3] = (data_norm * opacity * 255).astype(np.uint8)
    rgba[data_clean == 0, 3] = 0

    # Convert extent to Folium bounds
    lon_min, lat_min, lon_max, lat_max = extent
    bounds = [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]

    # Convert to PNG with fast compression
    img = Image.fromarray(rgba, mode="RGBA")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG", compress_level=1, optimize=False)
    img_buffer.seek(0)

    img_base64 = base64.b64encode(img_buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_base64}"

    # Add as ImageOverlay
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=1.0,  # Opacity already in RGBA
        name=name,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    return m


def add_geothermal_layer(
    m: folium.Map,
    data: pd.DataFrame,
    name: str,
    opacity: float = 0.7,
) -> folium.Map:
    """
    Add geothermal heat flow point data as markers.

    Parameters:
    -----------
    m : folium.Map
        Map to add layer to
    data : DataFrame
        Heat flow data with columns ['q', 'lat_NS', 'long_EW']
    name : str
        Layer name
    opacity : float
        Marker opacity (0-1)
    """
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap

    # Create color map (low heat = blue, high heat = red)
    cmap = cm.get_cmap("YlOrRd")

    # Normalize heat flow values (typical range 20-150 mW/m²)
    q_values = data["q"].values
    q_min, q_max = 20, 150
    q_norm = np.clip((q_values - q_min) / (q_max - q_min), 0, 1)

    # Create a feature group for the layer
    feature_group = folium.FeatureGroup(name=name)

    # Add points (sample if too many)
    max_points = 5000
    if len(data) > max_points:
        data_sample = data.sample(n=max_points, random_state=42)
        q_norm_sample = np.clip(
            (data_sample["q"].values - q_min) / (q_max - q_min), 0, 1
        )
    else:
        data_sample = data
        q_norm_sample = q_norm

    # Add circle markers
    import matplotlib.colors as mcolors

    for idx, (_, row) in enumerate(data_sample.iterrows()):
        color = mcolors.rgb2hex(cmap(q_norm_sample[idx])[:3])

        folium.CircleMarker(
            location=[row["lat_NS"], row["long_EW"]],
            radius=3,
            popup=f"Heat Flow: {row['q']:.1f} mW/m²",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=opacity,
            weight=0,
        ).add_to(feature_group)

    feature_group.add_to(m)
    return m


def add_geological_layer(
    m: folium.Map,
    data: Union[gpd.GeoDataFrame, Tuple[np.ndarray, tuple]],
    name: str,
    color: str,
    opacity: float,
) -> folium.Map:
    """
    Add geological storage layer to map.

    Parameters:
    -----------
    m : folium.Map
        Map to add layer to
    data : GeoDataFrame or tuple
        Either GeoDataFrame (vector) or (raster_array, extent) tuple
    name : str
        Layer name
    color : str
        Hex color code
    opacity : float
        Layer opacity (0-1)
    """
    # Check if data is rasterized (tuple) or vector (GeoDataFrame)
    if isinstance(data, tuple):
        # Rasterized data
        raster, extent = data
        add_raster_layer(m, raster, extent, name, color, opacity)
    else:
        # Vector data
        folium.GeoJson(
            data,
            name=name,
            style_function=lambda x: {
                "fillColor": color,
                "color": color,
                "weight": 0,
                "fillOpacity": opacity,
            },
        ).add_to(m)
    return m


def add_raster_layer(
    m: folium.Map,
    raster: np.ndarray,
    extent: tuple,
    name: str,
    color: str,
    opacity: float,
) -> folium.Map:
    """
    Add rasterized geological layer as ImageOverlay.

    Parameters:
    -----------
    m : folium.Map
        Map to add layer to
    raster : ndarray
        Boolean raster array in Web Mercator projection (EPSG:3857)
    extent : tuple
        (lon_min, lat_min, lon_max, lat_max) in geographic coordinates (EPSG:4326)
    name : str
        Layer name
    color : str
        Hex color code
    opacity : float
        Layer opacity (0-1)
    """
    import sys
    from pathlib import Path

    # Add project root to path if not already there
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from processing.rasterize import raster_to_image_overlay

    # Convert raster to RGBA image
    rgba, bounds = raster_to_image_overlay(raster, extent, color, opacity)

    # Convert numpy array to PNG bytes
    img = Image.fromarray(rgba, mode="RGBA")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    # Encode as base64 for folium
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_base64}"

    # Add as ImageOverlay
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=1.0,  # Opacity already baked into RGBA
        name=name,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    return m


def add_thickness_layer(
    m: folium.Map,
    data: np.ndarray,
    extent: tuple,
    name: str,
    thresholds: list,
    colors: list,
    opacity: float,
) -> folium.Map:
    """
    Add sedimentary thickness layer with graduated color scheme.

    Parameters:
    -----------
    m : folium.Map
        Map to add layer to
    data : ndarray
        Thickness values in meters
    extent : tuple
        (lon_min, lat_min, lon_max, lat_max) in geographic coordinates
    name : str
        Layer name
    thresholds : list
        Thickness thresholds in meters (e.g., [500, 1000, 2000, 5000, 10000])
    colors : list
        Hex color codes for each threshold bin
    opacity : float
        Layer opacity (0-1)
    """
    import time

    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import zoom

    start_time = time.time()
    print(
        f"[THICKNESS] Starting to add thickness layer. Data shape: {data.shape}",
        flush=True,
    )

    # Downsample large rasters to prevent browser performance issues
    MAX_DIM = 2000  # Maximum dimension for rendering
    height, width = data.shape

    if height > MAX_DIM or width > MAX_DIM:
        scale_factor = min(MAX_DIM / height, MAX_DIM / width)
        print(
            f"[THICKNESS] Downsampling from {data.shape} by factor {scale_factor:.3f}",
            flush=True,
        )

        # Use zoom for smooth downsampling, order=1 is bilinear interpolation
        data = zoom(data, scale_factor, order=1, mode="nearest")
        print(
            f"[THICKNESS] Downsampled to {data.shape} ({time.time() - start_time:.2f}s)",
            flush=True,
        )

    # Create graduated colormap from thresholds and colors
    # Add boundary colors for values below min and above max
    n_colors = len(colors)
    cmap = LinearSegmentedColormap.from_list("thickness", colors, N=256)
    print(f"[THICKNESS] Colormap created ({time.time() - start_time:.2f}s)", flush=True)

    # Normalize data to 0-1 range based on thresholds
    vmin, vmax = thresholds[0], thresholds[-1]
    data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    print(f"[THICKNESS] Data normalized ({time.time() - start_time:.2f}s)", flush=True)

    # Apply colormap
    rgba = cmap(data_norm)
    rgba = (rgba * 255).astype(np.uint8)
    print(f"[THICKNESS] Colormap applied ({time.time() - start_time:.2f}s)", flush=True)

    # Set transparency for NaN values and apply overall opacity
    mask_valid = ~np.isnan(data)
    rgba[:, :, 3] = np.where(mask_valid, int(opacity * 255), 0)
    print(f"[THICKNESS] Transparency set ({time.time() - start_time:.2f}s)", flush=True)

    # Convert extent to folium bounds
    lon_min, lat_min, lon_max, lat_max = extent
    bounds = [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]

    # Convert to PNG
    print(
        f"[THICKNESS] Converting to PNG image ({data.shape[0]}x{data.shape[1]} pixels)...",
        flush=True,
    )
    img = Image.fromarray(rgba, mode="RGBA")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG", compress_level=6)
    img_buffer.seek(0)
    print(
        f"[THICKNESS] PNG created, size: {len(img_buffer.getvalue()) / 1024 / 1024:.2f} MB ({time.time() - start_time:.2f}s)",
        flush=True,
    )

    print(f"[THICKNESS] Base64 encoding...", flush=True)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    print(
        f"[THICKNESS] Base64 size: {len(img_base64) / 1024 / 1024:.2f} MB ({time.time() - start_time:.2f}s)",
        flush=True,
    )
    img_url = f"data:image/png;base64,{img_base64}"

    # Add as ImageOverlay
    print(f"[THICKNESS] Adding ImageOverlay to map...", flush=True)
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=1.0,  # Opacity already in RGBA
        name=name,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    print(
        f"[THICKNESS] Thickness layer complete! Total time: {time.time() - start_time:.2f}s",
        flush=True,
    )
    return m
