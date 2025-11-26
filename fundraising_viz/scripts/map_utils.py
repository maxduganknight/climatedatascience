"""
Shared utilities for CDR map visualization scripts.

This module provides common constants, helper functions, and reusable components
for generating energy potential and storage capacity maps.
"""

import os

import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# =============================================================================
# SHARED REGION DEFINITIONS
# =============================================================================
# Path to world countries geojson (relative to scripts directory)
WORLD_COUNTRIES_GEOJSON = "../../data/shapefiles/downloaded/world_countries.geojson"

REGIONS = {
    "Canada": {"countries": ["Canada"], "continent": None},
    "Norway": {"countries": ["Norway"], "continent": None},
    "UK": {"countries": ["United Kingdom"], "continent": None},
    "Norway_UK": {"countries": ["United Kingdom", "Norway"], "continent": None},
    "Indonesia": {"countries": ["Indonesia"], "continent": None},
    "Saudi_Oman": {
        "countries": ["Saudi Arabia", "Oman"],
        "continent": None,
        "plot_buffer_degrees": 2.0,
    },
    "Australia": {"countries": ["Australia"], "continent": None},
    "Iceland": {
        "countries": ["Iceland"],
        "continent": None,
        "plot_buffer_degrees": 3.0,
    },
    "South Africa": {"countries": ["South Africa"], "continent": None},
    "US": {"countries": ["United States of America"], "continent": None},
    "Mexico": {"countries": ["Mexico"], "continent": None},
    "Brazil": {"countries": ["Brazil"], "continent": None},
    "Japan": {"countries": ["Japan"], "continent": None, "plot_buffer_degrees": 5.0},
}

# =============================================================================
# MAP STYLING CONSTANTS
# =============================================================================
MAP_STYLE = {
    "ocean_color": "#E8E8E8",
    "land_color": "#D0D0D0",
    "coastline_color": "#888888",
    "border_color": "#AAAAAA",
    "dpi": 900,
    "figsize_global": (20, 11),
    "figsize_regional": (16, 10),
    "icon_path": "/Users/max/Deep_Sky/design/Favicon/favicon_for_charts.png",
    "icon_zoom": 0.03,
}

# Storage colors for geological maps
STORAGE_COLORS = {
    "sedimentary": "#E9A12E",  # Warm tan/orange for sedimentary basins
    "basaltic": "#3F6B6F",  # Cool blue-gray for basaltic formations
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def add_base_features(ax, style="global"):
    """
    Add standard map features (land, ocean, coastlines, borders) to a map axis.

    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Map axis to add features to
    style : str
        'global' for world maps (thinner lines), 'regional' for regional maps (thicker lines)
    """
    ax.set_facecolor(MAP_STYLE["ocean_color"])
    ax.add_feature(cfeature.LAND, facecolor=MAP_STYLE["land_color"], zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=MAP_STYLE["ocean_color"], zorder=0)

    if style == "global":
        ax.add_feature(
            cfeature.COASTLINE,
            linewidth=0.5,
            edgecolor=MAP_STYLE["coastline_color"],
            zorder=5,
        )
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.3,
            edgecolor=MAP_STYLE["border_color"],
            linestyle=":",
            zorder=5,
        )
    else:  # regional
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#666666", zorder=5)
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.5,
            edgecolor="#888888",
            linestyle="-",
            zorder=5,
        )


def add_deepsky_icon(ax, icon_path=None):
    """
    Add Deep Sky icon to bottom-right corner of map.

    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Map axis to add icon to
    icon_path : str, optional
        Path to icon file. If None, uses MAP_STYLE['icon_path']
    """
    if icon_path is None:
        icon_path = MAP_STYLE["icon_path"]

    if not os.path.exists(icon_path):
        return

    icon = mpimg.imread(icon_path)
    imagebox = OffsetImage(icon, zoom=MAP_STYLE["icon_zoom"])
    ab = AnnotationBbox(
        imagebox,
        (0.95, 0.04),
        xycoords="figure fraction",
        frameon=False,
        box_alignment=(1.0, 0.0),
    )
    ax.add_artist(ab)


def save_map(fig, save_path, dpi=None):
    """
    Save figure as both PNG and SVG with consistent settings.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    save_path : str
        Path for PNG file (SVG will use same name with .svg extension)
    dpi : int, optional
        DPI for PNG output. If None, uses MAP_STYLE['dpi']
    """
    if not save_path:
        return

    if dpi is None:
        dpi = MAP_STYLE["dpi"]

    # Create directory if needed
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save PNG
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved PNG: {save_path}")

    # Save SVG
    svg_path = save_path.replace(".png", ".svg")
    plt.savefig(
        svg_path, format="svg", facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    print(f"Saved SVG: {svg_path}")


def create_land_mask(lon_grid, lat_grid, country_geom=None, resolution="50m"):
    """
    Create land mask for given coordinate grid.

    Parameters:
    -----------
    lon_grid, lat_grid : ndarray
        2D coordinate grids from np.meshgrid
    country_geom : shapely geometry, optional
        If provided, mask to country borders instead of all land
    resolution : str
        Natural Earth resolution ('110m', '50m', '10m')

    Returns:
    --------
    ndarray: Boolean mask where True = land (or inside country borders)
    """
    mask_shape = lon_grid.shape
    land_mask = np.zeros(mask_shape, dtype=bool)

    if country_geom is not None:
        # Country-specific masking
        if isinstance(country_geom, Polygon):
            polys = [country_geom]
        elif isinstance(country_geom, MultiPolygon):
            polys = list(country_geom.geoms)
        else:
            polys = [country_geom]
    else:
        # General land masking using Natural Earth
        land_feature = cfeature.NaturalEarthFeature("physical", "land", resolution)
        polys = []
        for geom in land_feature.geometries():
            if geom.geom_type == "Polygon":
                polys.append(geom)
            else:
                polys.extend(geom.geoms)

    # Apply masks from all polygons
    points_grid = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    for poly in polys:
        path = Path(np.array(poly.exterior.coords))
        mask_poly = path.contains_points(points_grid).reshape(mask_shape)
        land_mask |= mask_poly

    return land_mask


def load_countries_from_geojson(
    geojson_path, countries=None, continent=None, plot_buffer_degrees=0
):
    """
    Load country geometries from world_countries.geojson file.

    Parameters:
    -----------
    geojson_path : str
        Path to world_countries.geojson file
    countries : list of str, optional
        List of country names to load (e.g., ['Canada', 'United States of America'])
        If None, uses continent parameter
    continent : str, optional
        Continent name to load all countries from (e.g., 'Africa')
        Only used if countries is None
    plot_buffer_degrees : float, optional
        Buffer in degrees to add around country boundaries for plotting only.
        Does not affect data clipping or energy estimates. Useful for small
        countries like Iceland to zoom out the map view.

    Returns:
    --------
    tuple: (unified_geom, lat_range, lon_range, plot_lat_range, plot_lon_range)
        - unified_geom: Unified shapely geometry for all selected countries
        - lat_range: (min_lat, max_lat) - actual data bounds
        - lon_range: (min_lon, max_lon) - actual data bounds
        - plot_lat_range: (min_lat, max_lat) - bounds for plotting (with buffer)
        - plot_lon_range: (min_lon, max_lon) - bounds for plotting (with buffer)
    """
    gdf = gpd.read_file(geojson_path)

    # Ensure WGS84 projection
    if gdf.crs and gdf.crs != "EPSG:4326":
        print(f"  Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # Filter by countries or continent
    if countries is not None:
        # Select specific countries
        selected = gdf[gdf["admin"].isin(countries)]
        if len(selected) == 0:
            # Try fuzzy matching if exact match fails
            mask = gdf["admin"].apply(
                lambda x: any(c.lower() in x.lower() for c in countries)
            )
            selected = gdf[mask]
        print(f"  Selected {len(selected)} countries: {selected['admin'].tolist()}")
    elif continent is not None:
        # Select entire continent
        selected = gdf[gdf["continent"] == continent]
        print(f"  Selected {len(selected)} countries from {continent}")
    else:
        raise ValueError("Must provide either countries list or continent name")

    if len(selected) == 0:
        raise ValueError(
            f"No countries found matching criteria: countries={countries}, continent={continent}"
        )

    # Unify all selected geometries
    unified_geom = unary_union(selected.geometry)

    # Fix invalid geometries (important for complex multi-country regions like Africa)
    # The buffer(0) trick fixes many topology issues
    if not unified_geom.is_valid:
        print("  Fixing invalid geometry topology...")
        unified_geom = unified_geom.buffer(0)

    # Extract bounds
    bounds = selected.total_bounds  # [minx, miny, maxx, maxy]
    lat_range = (bounds[1], bounds[3])
    lon_range = (bounds[0], bounds[2])

    # Calculate plot bounds with buffer
    plot_lat_range = (
        lat_range[0] - plot_buffer_degrees,
        lat_range[1] + plot_buffer_degrees,
    )
    plot_lon_range = (
        lon_range[0] - plot_buffer_degrees,
        lon_range[1] + plot_buffer_degrees,
    )

    buffer_note = (
        f" (plot buffer: {plot_buffer_degrees}°)" if plot_buffer_degrees > 0 else ""
    )
    print(
        f"  Data bounds: lon({lon_range[0]:.2f}, {lon_range[1]:.2f}), "
        f"lat({lat_range[0]:.2f}, {lat_range[1]:.2f}){buffer_note}"
    )
    if plot_buffer_degrees > 0:
        print(
            f"  Plot bounds: lon({plot_lon_range[0]:.2f}, {plot_lon_range[1]:.2f}), "
            f"lat({plot_lat_range[0]:.2f}, {plot_lat_range[1]:.2f})"
        )

    return unified_geom, lat_range, lon_range, plot_lat_range, plot_lon_range


def load_country_shapefile(shapefile_path):
    """
    Load country shapefile and extract unified geometry with bounds.

    Parameters:
    -----------
    shapefile_path : str
        Path to shapefile

    Returns:
    --------
    tuple: (country_geom, lat_range, lon_range)
        - country_geom: Unified shapely geometry for the country
        - lat_range: (min_lat, max_lat)
        - lon_range: (min_lon, max_lon)
    """
    print(f"  Loading shapefile: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)

    # Ensure WGS84 projection
    if gdf.crs != "EPSG:4326":
        print(f"  Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # Unify all geometries
    country_geom = unary_union(gdf.geometry)

    # Extract bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    lat_range = (bounds[1], bounds[3])
    lon_range = (bounds[0], bounds[2])

    print(
        f"  Shapefile bounds: lon({lon_range[0]:.2f}, {lon_range[1]:.2f}), "
        f"lat({lat_range[0]:.2f}, {lat_range[1]:.2f})"
    )

    return country_geom, lat_range, lon_range


def sanitize_region_name(region_name):
    """
    Convert region name to filename-safe string.

    Parameters:
    -----------
    region_name : str
        Region name (e.g., "Norway & UK")

    Returns:
    --------
    str: Filename-safe name (e.g., "norway_uk")
    """
    return region_name.lower().replace(" & ", "_").replace(" ", "_")


def create_legend_item(
    fig, x, y, color, label, description, font_prop, square_size=0.015, alpha=0.7
):
    """
    Create a single legend item with colored square and text labels.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to add legend item to
    x, y : float
        Position in figure coordinates (0-1)
    color : str
        Color for the legend square
    label : str
        Main label text
    description : str
        Secondary description text (italic, smaller, gray)
    font_prop : FontProperties
        Font properties for text
    square_size : float
        Size of colored square in figure coordinates
    alpha : float
        Alpha transparency for colored square
    """
    # Create colored square
    rect = Rectangle(
        (x, y - 0.005),
        square_size,
        0.012,
        transform=fig.transFigure,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    fig.patches.append(rect)

    # Main label
    plt.figtext(
        x + square_size + 0.008,
        y,
        label,
        fontsize=11,
        fontproperties=font_prop,
        ha="left",
        va="center",
    )

    # Description
    plt.figtext(
        x + square_size + 0.008,
        y - 0.018,
        description,
        fontsize=9,
        fontproperties=font_prop,
        ha="left",
        va="center",
        color="#666666",
        style="italic",
    )
