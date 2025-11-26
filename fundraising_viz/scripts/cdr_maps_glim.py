"""
CDR Storage Potential Map using GLiM (Global Lithological Map) Data

This script creates a visualization of global CO2 storage potential based on
real geological data from Hartmann & Moosdorf (2012).

Data source: GLiM - Global Lithological Map
Paper: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012GC004370

Storage potential categories:
1. Basaltic/Ultramafic (Mineralization): vb, va, vi
2. Sedimentary (Porous storage): ss, sm, sc, su
"""

import os
import sys

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../reports")
from map_utils import (
    MAP_STYLE,
    REGIONS,
    STORAGE_COLORS,
    WORLD_COUNTRIES_GEOJSON,
    add_base_features,
    add_deepsky_icon,
    load_countries_from_geojson,
    load_country_shapefile,
    sanitize_region_name,
    save_map,
)
from utils import COLORS, setup_space_mono_font

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# GLiM lithology codes for CO2 storage potential
BASALTIC_CODES = [
    "vb",
    # 'va',
    "vi",
    "pb",
    "mt",
]  # Basaltic/volcanic for mineralization

SEDIMENTARY_CODES = [
    "ss"
    # ,'sm'
    # ,'sc'
    # #,'su'
]  # Sedimentary for porous storage


def load_glim_data(
    gdb_path="../data/cdr_mapper/storage/glim/LiMW_GIS 2015.gdb",
    simplify_tolerance=0.1,
    use_cache=True,
    cache_dir="data/co2_storage_analysis/cache",
):
    """
    Load and filter GLiM geological data for CDR storage potential.
    Uses caching to speed up subsequent loads.

    Parameters:
    -----------
    gdb_path : str
        Path to the GLiM geodatabase
    simplify_tolerance : float
        Tolerance for geometry simplification (in degrees). Higher = simpler/faster.
        Use 0.01-0.05 for balance between detail and performance.
    use_cache : bool
        If True, use cached data if available
    cache_dir : str
        Directory to store cached processed data

    Returns:
    --------
    dict with keys 'basaltic' and 'sedimentary', each containing a GeoDataFrame
    """
    print("=" * 60)
    print("Loading GLiM geological data...")
    print("=" * 60)

    # Generate cache filenames based on parameters
    basaltic_codes_str = "_".join(sorted(BASALTIC_CODES))
    sedimentary_codes_str = "_".join(sorted(SEDIMENTARY_CODES))
    cache_suffix = f"_tol{simplify_tolerance:.3f}"

    basaltic_cache = os.path.join(
        cache_dir, f"basaltic_{basaltic_codes_str}{cache_suffix}.gpkg"
    )
    sedimentary_cache = os.path.join(
        cache_dir, f"sedimentary_{sedimentary_codes_str}{cache_suffix}.gpkg"
    )

    # Check if cached files exist
    if (
        use_cache
        and os.path.exists(basaltic_cache)
        and os.path.exists(sedimentary_cache)
    ):
        print("\n✓ Found cached data, loading from cache...")
        print(f"  Basaltic cache: {basaltic_cache}")
        print(f"  Sedimentary cache: {sedimentary_cache}")

        basaltic_gdf = gpd.read_file(basaltic_cache)
        sedimentary_gdf = gpd.read_file(sedimentary_cache)

        print(f"\n  Loaded {len(basaltic_gdf):,} basaltic features")
        print(f"  Loaded {len(sedimentary_gdf):,} sedimentary features")
        print("\n✓ Cache loading complete (much faster!)")

        return {"basaltic": basaltic_gdf, "sedimentary": sedimentary_gdf}

    # No cache available, load from geodatabase
    print("\nNo cache found. Loading from geodatabase (this may take 2-5 minutes)...")
    gdf = gpd.read_file(gdb_path)

    print(f"  Loaded {len(gdf):,} geological features")
    print(f"  CRS: {gdf.crs}")

    # Reproject to WGS84 for easier plotting
    if gdf.crs != "EPSG:4326":
        print("\nReprojecting to WGS84...")
        gdf = gdf.to_crs("EPSG:4326")

    # Filter for basaltic/volcanic formations
    print(
        f"\nFiltering for basaltic/volcanic formations ({', '.join(BASALTIC_CODES)})..."
    )
    basaltic_gdf = gdf[gdf["xx"].isin(BASALTIC_CODES)].copy()
    print(f"  Found {len(basaltic_gdf):,} basaltic features")

    # Filter for sedimentary formations
    print(f"\nFiltering for sedimentary formations ({', '.join(SEDIMENTARY_CODES)})...")
    sedimentary_gdf = gdf[gdf["xx"].isin(SEDIMENTARY_CODES)].copy()
    print(f"  Found {len(sedimentary_gdf):,} sedimentary features")

    # Simplify geometries for faster rendering
    if simplify_tolerance > 0:
        print(f"\nSimplifying geometries (tolerance={simplify_tolerance})...")
        basaltic_gdf["geometry"] = basaltic_gdf["geometry"].simplify(simplify_tolerance)
        sedimentary_gdf["geometry"] = sedimentary_gdf["geometry"].simplify(
            simplify_tolerance
        )
        print("  ✓ Geometries simplified")

    # Save to cache for future use
    print(f"\nSaving processed data to cache...")
    os.makedirs(cache_dir, exist_ok=True)

    basaltic_gdf.to_file(basaltic_cache, driver="GPKG")
    sedimentary_gdf.to_file(sedimentary_cache, driver="GPKG")

    print(f"  ✓ Saved basaltic data: {basaltic_cache}")
    print(f"  ✓ Saved sedimentary data: {sedimentary_cache}")
    print("  (Future runs will be much faster!)")

    print("\n✓ Data loading complete")

    return {"basaltic": basaltic_gdf, "sedimentary": sedimentary_gdf}


def create_cdr_storage_map(geological_data, save_path=None):
    """
    Create a world map showing CO2 storage potential based on geology.

    Parameters:
    -----------
    geological_data : dict
        Dictionary with 'basaltic' and 'sedimentary' GeoDataFrames
    save_path : str, optional
        Path to save the figure
    """
    print("\n" + "=" * 60)
    print("Creating CDR storage potential map...")
    print("=" * 60)

    # Setup font
    font_props = setup_space_mono_font()
    font_prop = font_props.get("regular") if font_props else None

    # Create figure with specific size
    fig = plt.figure(
        figsize=MAP_STYLE["figsize_global"], facecolor=COLORS["background"]
    )

    # Create map with Robinson projection (good for world maps)
    ax = plt.axes(projection=ccrs.Robinson())

    # Add base map features
    add_base_features(ax, style="global")

    # Set global extent
    ax.set_global()

    # Plot sedimentary formations first (lower layer)
    print("\nPlotting sedimentary storage potential...")
    sedimentary_gdf = geological_data["sedimentary"]
    if len(sedimentary_gdf) > 0:
        sedimentary_gdf.plot(
            ax=ax,
            color=STORAGE_COLORS["sedimentary"],
            alpha=0.8,
            edgecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        print(f"  ✓ Plotted {len(sedimentary_gdf):,} sedimentary features")

    # Plot basaltic formations on top
    print("\nPlotting basaltic/volcanic storage potential...")
    basaltic_gdf = geological_data["basaltic"]
    if len(basaltic_gdf) > 0:
        basaltic_gdf.plot(
            ax=ax,
            color=STORAGE_COLORS["basaltic"],
            alpha=0.8,
            edgecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        print(f"  ✓ Plotted {len(basaltic_gdf):,} basaltic features")

    # Add title and subtitle
    # plt.figtext(0.5, 0.96, 'GLOBAL CO₂ STORAGE POTENTIAL FOR CDR',
    #            fontsize=22, fontweight='bold',
    #            fontproperties=title_font,
    #            ha='center', va='center')

    plt.figtext(
        0.5,
        0.92,
        "Geological formations with potential for carbon dioxide storage",
        fontsize=14,
        fontproperties=font_prop,
        ha="center",
        va="center",
        color="#444444",
    )

    # Create legend
    from map_utils import create_legend_item

    legend_x = 0.12
    legend_y = 0.45
    y_offset = 0.04

    # Sedimentary storage
    create_legend_item(
        fig,
        legend_x,
        legend_y - y_offset,
        STORAGE_COLORS["sedimentary"],
        "Sedimentary Basins",
        "Sequestration potential in deep saline aquifers",
        font_prop,
        alpha=0.6,
    )

    # Basaltic storage
    create_legend_item(
        fig,
        legend_x,
        legend_y - y_offset * 2.5,
        STORAGE_COLORS["basaltic"],
        "Basaltic & Ultramafic Formations",
        "In-situ mineralization potential in metamorphosed\nmafic/ultramafic rocks.",
        font_prop,
        alpha=0.6,
    )

    # Add data note and attribution
    data_note = "DATA: HARTMANN & MOOSDORF (2012), GLOBAL LITHOLOGICAL MAP (GLiM).\nBasaltic & Ultramafic includes basaltic volcanics, intermediate volcanics, mafic plutonics, and metamorphic rocks which may include serpentinized ultramafics."

    plt.figtext(
        0.12,
        0.04,
        data_note,
        fontsize=9,
        color="#505050",
        ha="left",
        va="bottom",
        fontproperties=font_prop,
    )

    # Add Deep Sky icon
    add_deepsky_icon(ax)

    # Adjust layout to center the map and maximize space
    plt.subplots_adjust(left=0.01, right=1.02, top=0.88, bottom=0.08)

    # Save if path provided
    save_map(fig, save_path)

    print("\n✓ Map creation complete")
    return fig


def create_regional_storage_map(
    geological_data,
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
    Create a regional map showing CO2 storage potential based on geology.

    Parameters:
    -----------
    geological_data : dict
        Dictionary with 'basaltic' and 'sedimentary' GeoDataFrames
    region_name : str
        Name of the region for the title
    lat_range : tuple
        (min_lat, max_lat) for data clipping (actual country bounds)
    lon_range : tuple
        (min_lon, max_lon) for data clipping (actual country bounds)
    save_path : str, optional
        Path to save the figure
    country_shapefile : str, optional
        Path to shapefile for country-specific masking (e.g., Canada)
    country_geom : shapely geometry, optional
        Pre-loaded country/region geometry. Takes precedence over country_shapefile.
    plot_lat_range : tuple, optional
        (min_lat, max_lat) for plot extent. If None, uses lat_range.
    plot_lon_range : tuple, optional
        (min_lon, max_lon) for plot extent. If None, uses lon_range.
    """
    print("\n" + "=" * 60)
    print(f"Creating regional CDR storage map for {region_name}...")
    print("=" * 60)

    # Setup font
    font_props = setup_space_mono_font()
    font_prop = font_props.get("regular") if font_props else None

    # Load country shapefile if provided and no geometry passed
    if country_geom is None and country_shapefile:
        country_geom, lat_range, lon_range = load_country_shapefile(country_shapefile)

    # Use plot bounds if provided, otherwise use data bounds
    if plot_lat_range is None:
        plot_lat_range = lat_range
    if plot_lon_range is None:
        plot_lon_range = lon_range

    # Clip geological data to region bounds (use data bounds, not plot bounds)
    print(f"\nClipping geological data to region...")
    basaltic_gdf = (
        geological_data["basaltic"]
        .cx[lon_range[0] : lon_range[1], lat_range[0] : lat_range[1]]
        .copy()
    )
    sedimentary_gdf = (
        geological_data["sedimentary"]
        .cx[lon_range[0] : lon_range[1], lat_range[0] : lat_range[1]]
        .copy()
    )

    print(f"  Basaltic features in region: {len(basaltic_gdf):,}")
    print(f"  Sedimentary features in region: {len(sedimentary_gdf):,}")

    # If country geometry provided, clip to country borders
    if country_geom is not None:
        print(f"\nClipping to country borders...")

        # Fix invalid geometries before clipping (common with complex multi-country boundaries)
        # Use buffer(0) trick to fix topology issues
        try:
            if not country_geom.is_valid:
                print("  Fixing invalid country geometry...")
                country_geom = country_geom.buffer(0)
        except Exception as e:
            print(f"  WARNING: Could not validate country geometry: {e}")

        if len(basaltic_gdf) > 0:
            try:
                basaltic_gdf = gpd.clip(basaltic_gdf, country_geom)
            except Exception as e:
                print(f"  WARNING: Could not clip basaltic features: {e}")
                print(f"  Attempting with geometry repair...")
                try:
                    # Try to repair the clip geometry
                    basaltic_gdf["geometry"] = basaltic_gdf["geometry"].buffer(0)
                    basaltic_gdf = gpd.clip(basaltic_gdf, country_geom)
                except Exception as e2:
                    print(f"  ERROR: Still failed after repair: {e2}")
                    print(f"  Skipping basaltic clipping for this region")

        if len(sedimentary_gdf) > 0:
            try:
                sedimentary_gdf = gpd.clip(sedimentary_gdf, country_geom)
            except Exception as e:
                print(f"  WARNING: Could not clip sedimentary features: {e}")
                print(f"  Attempting with geometry repair...")
                try:
                    # Try to repair the clip geometry
                    sedimentary_gdf["geometry"] = sedimentary_gdf["geometry"].buffer(0)
                    sedimentary_gdf = gpd.clip(sedimentary_gdf, country_geom)
                except Exception as e2:
                    print(f"  ERROR: Still failed after repair: {e2}")
                    print(f"  Skipping sedimentary clipping for this region")

        print(f"  Basaltic features after clipping: {len(basaltic_gdf):,}")
        print(f"  Sedimentary features after clipping: {len(sedimentary_gdf):,}")

    # Create figure
    fig = plt.figure(
        figsize=MAP_STYLE["figsize_global"], facecolor=COLORS["background"]
    )

    # Create map with PlateCarree projection (better for regional maps)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent (use plot bounds which may include buffer)
    ax.set_extent(
        [plot_lon_range[0], plot_lon_range[1], plot_lat_range[0], plot_lat_range[1]],
        crs=ccrs.PlateCarree(),
    )

    # Add base map features
    add_base_features(ax, style="regional")

    # Plot sedimentary formations first (lower layer)
    print("\nPlotting sedimentary storage potential...")
    if len(sedimentary_gdf) > 0:
        sedimentary_gdf.plot(
            ax=ax,
            color=STORAGE_COLORS["sedimentary"],
            alpha=0.8,
            edgecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        print(f"  ✓ Plotted {len(sedimentary_gdf):,} sedimentary features")
    else:
        print("  No sedimentary features in region")

    # Plot basaltic formations on top
    print("\nPlotting basaltic/volcanic storage potential...")
    if len(basaltic_gdf) > 0:
        basaltic_gdf.plot(
            ax=ax,
            color=STORAGE_COLORS["basaltic"],
            alpha=0.6,
            edgecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        print(f"  ✓ Plotted {len(basaltic_gdf):,} basaltic features")
    else:
        print("  No basaltic features in region")

    # Add subtitle
    plt.figtext(
        0.5,
        0.92,
        f"{region_name}: Geological formations with CO₂ storage potential",
        fontsize=14,
        fontproperties=font_prop,
        ha="center",
        va="center",
        color="#444444",
    )

    # Add data note
    data_note = "DATA: HARTMANN & MOOSDORF (2012), GLOBAL LITHOLOGICAL MAP (GLiM)."
    plt.figtext(
        0.12,
        0.04,
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
    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.08)

    # Save if path provided
    save_map(fig, save_path)

    print("\n✓ Regional map creation complete")
    return fig


def main():
    """
    Main function to generate CDR storage potential map from GLiM data.
    """

    print("\n" + "=" * 70)
    print("CDR STORAGE POTENTIAL VISUALIZATION")
    print("Using GLiM (Global Lithological Map) Real Geological Data")
    print("=" * 70)

    # Load geological data
    gdb_path = "data/co2_storage_analysis/LiMW_GIS 2015.gdb"

    # Load with moderate simplification for balance of detail and speed
    # Adjust simplify_tolerance if needed: 0.01 (detailed) to 0.1 (fast)
    geological_data = load_glim_data(gdb_path, simplify_tolerance=0.3)

    # Create global map
    save_path = "figures/cdr_storage_potential_glim.png"
    create_cdr_storage_map(geological_data, save_path=save_path)

    # =========================================================================
    # GENERATE REGIONAL MAPS
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING REGIONAL STORAGE MAPS")
    print("=" * 70)

    # Generate regional maps using shared region definitions
    for region_name, region_config in REGIONS.items():
        print(f"\n{'=' * 70}")
        print(f"Region: {region_name}")
        print(f"{'=' * 70}")

        # Load country/continent geometry
        script_dir = os.path.dirname(os.path.abspath(__file__))
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
            # Use loaded bounds from actual country geometry
            country_shapefile = None
        except Exception as e:
            print(f"  ERROR: Could not load geometry for {region_name}: {e}")
            print(f"  Skipping this region")
            continue

        # Create regional storage map
        region_filename = sanitize_region_name(region_name)
        storage_save_path = f"figures/regional/cdr_storage_{region_filename}.png"
        fig_storage = create_regional_storage_map(
            geological_data,
            region_name=region_name,
            lat_range=lat_range,
            lon_range=lon_range,
            save_path=storage_save_path,
            country_shapefile=country_shapefile,
            country_geom=country_geom,
            plot_lat_range=plot_lat_range,
            plot_lon_range=plot_lon_range,
        )
        plt.close(fig_storage)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nGlobal map output: {save_path}")
    print(f"Regional maps output: figures/regional/")
    print("\nKey insights:")
    print(f"  - Sedimentary basins: {len(geological_data['sedimentary']):,} formations")
    print(f"  - Basaltic formations: {len(geological_data['basaltic']):,} formations")
    print("\nLimitations:")
    print("  - Shows surface geology only (not subsurface depth)")
    print("  - Actual storage requires formations >800m depth")
    print("  - Site-specific assessment needed for any deployment")
    print("  - Does not account for existing infrastructure or accessibility")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
