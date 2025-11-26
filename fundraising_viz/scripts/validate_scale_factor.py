#!/usr/bin/env python3
"""
Validate the 0.8 scale factor across multiple years
"""
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import os
import glob
from shapely import affinity
from rasterio.transform import from_bounds
from rasterio.features import rasterize

def calculate_emissions_with_scale_factor(filepath, scale_factor=1.0):
    """Calculate emissions for a single year with given scale factor"""

    ds = xr.open_dataset(filepath)

    # Canada setup
    canada_gdf = gpd.read_file('data/emissions_gap/ca_shp/ca.shp')
    canada_gdf_360 = canada_gdf.copy()
    canada_gdf_360['geometry'] = canada_gdf_360['geometry'].apply(
        lambda geom: affinity.translate(geom, xoff=360)
    )
    bounds = canada_gdf_360.total_bounds

    # Subset data
    ds_canada = ds.sel(
        longitude=slice(bounds[0], bounds[2]),
        latitude=slice(bounds[3], bounds[1])
    )

    # Create Canada mask
    transform = from_bounds(
        ds_canada.longitude.min().values, ds_canada.latitude.min().values,
        ds_canada.longitude.max().values, ds_canada.latitude.max().values,
        len(ds_canada.longitude), len(ds_canada.latitude)
    )

    canada_mask = rasterize(
        canada_gdf_360.geometry,
        out_shape=(len(ds_canada.latitude), len(ds_canada.longitude)),
        transform=transform, fill=0, default_value=1, dtype='uint8'
    ).astype(bool)

    # Calculate grid cell areas
    lat_res = float(np.abs(ds_canada.latitude.diff('latitude').mean()))
    lon_res = float(np.abs(ds_canada.longitude.diff('longitude').mean()))
    dlat = np.radians(lat_res)
    dlon = np.radians(lon_res)
    earth_radius = 6371000

    lat_2d, lon_2d = np.meshgrid(ds_canada.latitude.values, ds_canada.longitude.values, indexing='ij')
    lat_rad_2d = np.radians(lat_2d)
    cell_areas = earth_radius**2 * dlat * dlon * np.cos(lat_rad_2d)

    # Calculate emissions with scale factor
    emissions_masked = ds_canada.co2fire.values * canada_mask[np.newaxis, :, :] * cell_areas[np.newaxis, :, :]
    daily_totals = np.nansum(emissions_masked, axis=(1, 2)) * 86400
    total_emissions = np.sum(daily_totals) / 1e9 * scale_factor

    ds.close()
    return total_emissions

def main():
    print("Validating 0.8 scale factor across available years")
    print("="*60)

    # Get available CAMS files
    cams_files = glob.glob('data/wildfire_emissions/cams_wildfire_emissions_*.nc')
    cams_files.sort()

    if not cams_files:
        print("No CAMS files found!")
        return

    # GWIS reference data (from our earlier analysis)
    gwis_data = {
        2005: 134.6,
        2006: 181.8,
        2007: 175.0,
        2008: 106.9
    }

    # Test different scale factors
    scale_factors = [0.7, 0.75, 0.8, 0.85, 0.9]

    results = {}
    for scale_factor in scale_factors:
        results[scale_factor] = {}

    print("Processing years:")

    for filepath in cams_files:
        filename = os.path.basename(filepath)
        year = int(filename.split('_')[-1].replace('.nc', ''))

        if year not in gwis_data:
            print(f"  Skipping {year} (no GWIS reference data)")
            continue

        print(f"  Processing {year}...")

        gwis_value = gwis_data[year]

        for scale_factor in scale_factors:
            try:
                emissions = calculate_emissions_with_scale_factor(filepath, scale_factor)
                diff_pct = (emissions - gwis_value) / gwis_value * 100
                results[scale_factor][year] = {
                    'emissions': emissions,
                    'gwis': gwis_value,
                    'diff_pct': diff_pct
                }
            except Exception as e:
                print(f"    Error with scale factor {scale_factor}: {e}")
                continue

    # Analysis
    print(f"\n{'Year':<6} {'GWIS':<8} " + "".join([f"SF {sf:<8}" for sf in scale_factors]))
    print("-" * (6 + 8 + len(scale_factors) * 10))

    for year in sorted(gwis_data.keys()):
        if any(year in results[sf] for sf in scale_factors):
            row = f"{year:<6} {gwis_data[year]:<8.1f} "
            for sf in scale_factors:
                if year in results[sf]:
                    emissions = results[sf][year]['emissions']
                    row += f"{emissions:<9.1f} "
                else:
                    row += f"{'--':<9} "
            print(row)

    # Summary statistics
    print(f"\nAverage absolute percentage errors:")
    print(f"{'Scale Factor':<12} {'Mean |Error|':<12} {'Std Error':<12} {'Max |Error|':<12}")
    print("-" * 48)

    best_scale_factor = None
    best_mean_error = float('inf')

    for sf in scale_factors:
        if not results[sf]:
            continue

        errors = [abs(data['diff_pct']) for data in results[sf].values()]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)

        print(f"{sf:<12.2f} {mean_error:<12.1f} {std_error:<12.1f} {max_error:<12.1f}")

        if mean_error < best_mean_error:
            best_mean_error = mean_error
            best_scale_factor = sf

    print(f"\nBest scale factor: {best_scale_factor} (mean absolute error: {best_mean_error:.1f}%)")

    # Detailed breakdown for best scale factor
    if best_scale_factor and results[best_scale_factor]:
        print(f"\nDetailed results for scale factor {best_scale_factor}:")
        print(f"{'Year':<6} {'CAMS (Mt)':<12} {'GWIS (Mt)':<12} {'Diff (Mt)':<12} {'% Error':<10}")
        print("-" * 60)

        for year in sorted(results[best_scale_factor].keys()):
            data = results[best_scale_factor][year]
            emissions = data['emissions']
            gwis = data['gwis']
            diff_mt = emissions - gwis
            diff_pct = data['diff_pct']
            print(f"{year:<6} {emissions:<12.1f} {gwis:<12.1f} {diff_mt:<12.1f} {diff_pct:<10.1f}%")

if __name__ == "__main__":
    main()