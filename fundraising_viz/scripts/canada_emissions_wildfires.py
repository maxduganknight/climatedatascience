import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import xarray as xr
import geopandas as gpd
import numpy as np
import datetime

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot

# Import emissions gap annotation function
sys.path.append('.')
from emissions_gap import annotate_emissions_gap

# Divide CAMS GFAS emissions sums by this calibration factor to resolve discrepancy between raw CAMS data and preprocessed GWIS data from eg: https://gwis.jrc.ec.europa.eu/apps/country.profile/charts/emi
# The GFAS paper discusses this discrepancy - the dry matter burned in GFAS is 1.9-2.2 times higher than GFED estimates
# https://bg.copernicus.org/articles/9/527/2012/bg-9-527-2012.pdf
GWIS_CALIBRATION_FACTOR = 0.85

def load_all_cams_wildfire_emissions(data_dir, shapefile_path, cache_dir='data/wildfire_emissions'):
    """
    Load and process all CAMS wildfire emissions data for Canada from 2005.
    Each file contains 1 year of data based on filename (e.g., 2005.nc has 2005 data).
    Uses CSV cache to speed up subsequent runs.
    Returns a DataFrame with year and wildfire_emissions columns.
    """
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    from shapely import affinity
    import glob
    import os
    import re

    # Check for cached CSV files first
    cache_file = os.path.join(cache_dir, 'canada_cams_wildfire_emissions.csv')
    provincial_cache_file = os.path.join(cache_dir, 'canada_cams_wildfire_emissions_provincial.csv')

    if os.path.exists(cache_file):
        print(f"Loading cached wildfire emissions data from {cache_file}")
        return pd.read_csv(cache_file)

    print("No cache found. Processing CAMS NetCDF files...")

    # Get all CAMS files
    cams_files = glob.glob(os.path.join(data_dir, 'cams_wildfire_emissions_*.nc'))
    cams_files.sort()

    print(f"Processing {len(cams_files)} CAMS files for Canada wildfire emissions...")

    # Load Canada shapefile once
    canada_gdf = gpd.read_file(shapefile_path)
    canada_gdf_360 = canada_gdf.copy()
    canada_gdf_360['geometry'] = canada_gdf_360['geometry'].apply(
        lambda geom: affinity.translate(geom, xoff=360)
    )
    bounds = canada_gdf_360.total_bounds

    yearly_emissions = {}
    provincial_emissions = []

    # Process each file
    for filepath in cams_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        try:
            # Extract year from filename (e.g., "2005" from "cams_wildfire_emissions_2005.nc")
            year_match = re.search(r'(\d{4})\.nc$', filename)
            if year_match:
                year = int(year_match.group(1))
                years_to_process = [year]
            else:
                print(f"  Could not parse year from {filename}")
                continue

            # Load the dataset once
            ds = xr.open_dataset(filepath)
            # Subset to Canada region once
            ds_canada = ds.sel(
                longitude=slice(bounds[0], bounds[2]),
                latitude=slice(bounds[3], bounds[1])
            )

            # Create Canada mask once per file
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

            # Calculate grid cell areas once per file
            _, lat_grid = np.meshgrid(ds_canada.longitude.values, ds_canada.latitude.values)
            lat_rad = np.radians(lat_grid)
            cell_areas = (6371000 ** 2) * np.radians(0.1) ** 2 * np.cos(lat_rad)

            # Process the year in this file
            year = years_to_process[0]

            # Calculate national emissions for this year
            emissions_masked = ds_canada.co2fire.values * canada_mask * cell_areas
            daily_totals = np.nansum(emissions_masked, axis=(1, 2)) * 86400
            total_emissions_mt = np.sum(daily_totals) / 1e9 * GWIS_CALIBRATION_FACTOR

            yearly_emissions[year] = total_emissions_mt

            # Calculate provincial emissions for this year
            for _, province in canada_gdf_360.iterrows():
                province_mask = rasterize(
                    [province.geometry],
                    out_shape=(len(ds_canada.latitude), len(ds_canada.longitude)),
                    transform=transform, fill=0, default_value=1, dtype='uint8'
                ).astype(bool)

                province_emissions_masked = ds_canada.co2fire.values * province_mask * cell_areas
                province_daily_totals = np.nansum(province_emissions_masked, axis=(1, 2)) * 86400
                province_total_mt = np.sum(province_daily_totals) / 1e9 * GWIS_CALIBRATION_FACTOR

                # Remove accents from province name for clean CSV output
                import unicodedata
                province_name = unicodedata.normalize('NFD', province['name']).encode('ascii', 'ignore').decode('ascii')

                provincial_emissions.append({
                    'year': year,
                    'province': province_name,
                    'wildfire_emissions': province_total_mt
                })

            ds.close()

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    # Convert to DataFrames
    wildfire_df = pd.DataFrame([
        {'year': year, 'wildfire_emissions': emissions}
        for year, emissions in yearly_emissions.items()
    ])
    wildfire_df = wildfire_df.sort_values('year').reset_index(drop=True)

    provincial_df = pd.DataFrame(provincial_emissions)
    provincial_df = provincial_df.sort_values(['year', 'province']).reset_index(drop=True)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    wildfire_df.to_csv(cache_file, index=False)
    provincial_df.to_csv(provincial_cache_file, index=False)
    print(f"Saved processed data to cache: {cache_file}")
    print(f"Saved provincial data to cache: {provincial_cache_file}")

    return wildfire_df


def load_brazil_wildfire_emissions(data_dir, shapefile_path, cache_dir='data/wildfire_emissions'):
    """
    Load and process all CAMS wildfire emissions data for Brazil from 2003.
    Uses the same approach as load_all_cams_wildfire_emissions but for Brazil.
    Uses CSV cache to speed up subsequent runs.
    Returns a DataFrame with year and wildfire_emissions columns.
    """
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    from shapely import affinity
    import glob
    import os
    import re

    # Check for cached CSV file first
    cache_file = os.path.join(cache_dir, 'brazil_cams_wildfire_emissions.csv')

    if os.path.exists(cache_file):
        print(f"Loading cached Brazil wildfire emissions data from {cache_file}")
        return pd.read_csv(cache_file)

    print("No cache found. Processing CAMS NetCDF files for Brazil...")

    # Get all CAMS files
    cams_files = glob.glob(os.path.join(data_dir, 'cams_wildfire_emissions_*.nc'))
    cams_files.sort()

    print(f"Processing {len(cams_files)} CAMS files for Brazil wildfire emissions...")

    # Load Brazil shapefile
    brazil_gdf = gpd.read_file(shapefile_path)

    # Brazil is in western hemisphere with negative longitudes, but CAMS uses 0-360
    # Shift Brazil geometry to 0-360 longitude system
    brazil_gdf_360 = brazil_gdf.copy()
    brazil_gdf_360['geometry'] = brazil_gdf_360['geometry'].apply(
        lambda geom: affinity.translate(geom, xoff=360)
    )
    bounds = brazil_gdf_360.total_bounds

    yearly_emissions = {}

    # Process each file
    for filepath in cams_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        try:
            # Extract year from filename
            year_match = re.search(r'(\d{4})\.nc$', filename)
            if year_match:
                year = int(year_match.group(1))
            else:
                print(f"  Could not parse year from {filename}")
                continue

            # Load the dataset
            ds = xr.open_dataset(filepath)

            # Subset to Brazil region
            ds_brazil = ds.sel(
                longitude=slice(bounds[0], bounds[2]),
                latitude=slice(bounds[3], bounds[1])
            )

            # Create Brazil mask
            transform = from_bounds(
                ds_brazil.longitude.min().values, ds_brazil.latitude.min().values,
                ds_brazil.longitude.max().values, ds_brazil.latitude.max().values,
                len(ds_brazil.longitude), len(ds_brazil.latitude)
            )

            brazil_mask = rasterize(
                brazil_gdf_360.geometry,
                out_shape=(len(ds_brazil.latitude), len(ds_brazil.longitude)),
                transform=transform, fill=0, default_value=1, dtype='uint8'
            ).astype(bool)

            # Calculate grid cell areas
            _, lat_grid = np.meshgrid(ds_brazil.longitude.values, ds_brazil.latitude.values)
            lat_rad = np.radians(lat_grid)
            cell_areas = (6371000 ** 2) * np.radians(0.1) ** 2 * np.cos(lat_rad)

            # Calculate national emissions for this year
            emissions_masked = ds_brazil.co2fire.values * brazil_mask * cell_areas
            daily_totals = np.nansum(emissions_masked, axis=(1, 2)) * 86400
            total_emissions_mt = np.sum(daily_totals) / 1e9 * GWIS_CALIBRATION_FACTOR

            yearly_emissions[year] = total_emissions_mt

            ds.close()

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    # Convert to DataFrame
    wildfire_df = pd.DataFrame([
        {'year': year, 'wildfire_emissions': emissions}
        for year, emissions in yearly_emissions.items()
    ])
    wildfire_df = wildfire_df.sort_values('year').reset_index(drop=True)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    wildfire_df.to_csv(cache_file, index=False)
    print(f"Saved Brazil data to cache: {cache_file}")

    return wildfire_df


def load_emissions_data(path):
    """
    Load Canada emissions data from 440 Megatonnes data provided here: https://dashboard.440megatonnes.ca/?_gl=1*1vk8sh9*_gcl_au*MTUyNzgzNDIzNy4xNzU4NDY3MTAx*_ga*MTA5NTAyMzkzNC4xNzU4NDY3MTAw*_ga_DVTX0HL4Z5*czE3NTg0NjcxMDAkbzEkZzEkdDE3NTg0NjczNTEkajYwJGwwJGgw
    The early estimate of 2024 emissions comes from here: https://440megatonnes.ca/early-estimate-of-national-emissions/
    The numbers on the second page are higher than the first page, but we don't get all years just the IPCC selected years. 
    So to add 2024 data I took the % change in emissions from the second page and applied it to the first page's 2023 emissions total. 
    """
    df = pd.read_excel(path, sheet_name='Data')

    # Filter for national sector data
    national_data = df[df['sector'] == 'national'].copy()

    # Filter for years 2005-2023 (historical data)
    historical_data = national_data[(national_data['year'] >= 2005) & (national_data['year'] <= 2023)]

    # Select only year and ghg columns, and ensure unique years
    emissions_df = historical_data[['year', 'ghg']].copy()
    emissions_df.rename(columns={'ghg': 'emissions'}, inplace=True)
    emissions_df = emissions_df.drop_duplicates(subset=['year']).reset_index(drop=True)

    # Add 2024 value (0.1% higher than 2023)
    emissions_2023 = emissions_df[emissions_df['year'] == 2023]['emissions'].iloc[0]
    emissions_2024 = emissions_2023 * 1.001  # 0.1% increase

    # Add 2024 row
    new_row = pd.DataFrame({'year': [2024], 'emissions': [emissions_2024]})
    emissions_df = pd.concat([emissions_df, new_row], ignore_index=True)

    # Load NZP pathway data (2025-2050)
    nzp_data = national_data[national_data['scenario'].isin(['NZP_lower', 'NZP_upper'])]
    nzp_data = nzp_data[(nzp_data['year'] >= 2025) & (nzp_data['year'] <= 2050)]

    # Pivot NZP data to get upper and lower bounds
    nzp_pivot = nzp_data.pivot(index='year', columns='scenario', values='ghg')
    nzp_pivot = nzp_pivot.reset_index()
    nzp_pivot.columns.name = None
    print(emissions_df.head())
    print(nzp_pivot.head())
    return emissions_df, nzp_pivot

def merge_emissions_with_wildfire(emissions_df, wildfire_df):
    """
    Merge historical emissions data with CAMS wildfire emissions data.
    Creates a combined emissions column for all available years.
    If the most recent year is missing the national emissions data 
    (because I can get CAMS wildfire emissions data before national emissions data)
    then use the previous year's emissions data and sum with the latest year's wildfire emissions data.
    This is done by forward filling nans in the sum line at the bottom.
    """
    # Merge the dataframes on year
    merged_df = emissions_df.merge(wildfire_df, on='year', how='right')

    # Fill NaN wildfire emissions with 0 for years that don't have wildfire data
    merged_df['wildfire_emissions'] = merged_df['wildfire_emissions'].fillna(0)

    # Create combined emissions column forward filling Nans
    merged_df['combined_emissions'] = merged_df['emissions'].ffill() + merged_df['wildfire_emissions']

    return merged_df


def calculate_baseline_average(wildfire_df, start_year=2003, end_year=2018):
    """
    Calculate the baseline average for Canada wildfire emissions from 2003-2018 (15 years).
    """
    baseline_data = wildfire_df[
        (wildfire_df['year'] >= start_year) &
        (wildfire_df['year'] <= end_year)
    ]

    baseline_average = baseline_data['wildfire_emissions'].mean()
    print(f"Baseline average wildfire emissions ({start_year}-{end_year}): {baseline_average:.1f} Mt")

    return baseline_average

def calculate_emissions_anomaly(wildfire_df, baseline_average):
    """
    Calculate yearly emissions anomaly relative to the baseline average.
    """
    df_with_anomaly = wildfire_df.copy()
    df_with_anomaly['emissions_anomaly'] = df_with_anomaly['wildfire_emissions'] - baseline_average

    return df_with_anomaly

def calculate_rolling_anomaly_average(df_with_anomaly, window=10):
    """
    Calculate rolling X-year average of emissions anomaly.
    """
    df_with_rolling = df_with_anomaly.copy()
    df_with_rolling['emissions_anomaly_3y_average'] = df_with_rolling['emissions_anomaly'].rolling(
        window=window, center=True, min_periods=1
    ).mean()

    return df_with_rolling

def merge_emissions_with_anomaly_adjusted_wildfires(emissions_df, wildfire_anomaly_df):
    """
    Merge historical emissions data with anomaly-adjusted wildfire emissions.
    Creates combined emissions using the rolling 3-year anomaly average.
    """
    # Merge the dataframes on year
    merged_df = emissions_df.merge(wildfire_anomaly_df, on='year', how='right')

    # Fill NaN wildfire emissions and anomaly values with 0 for years that don't have wildfire data
    merged_df['wildfire_emissions'] = merged_df['wildfire_emissions'].fillna(0)
    merged_df['emissions_anomaly'] = merged_df['emissions_anomaly'].fillna(0)
    merged_df['emissions_anomaly_3y_average'] = merged_df['emissions_anomaly_3y_average'].fillna(0)

    # Create anomaly-adjusted combined emissions using 3-year rolling average
    merged_df['anomaly_adjusted_emissions'] = (
        merged_df['emissions'].ffill() + merged_df['emissions_anomaly_3y_average']
    )

    return merged_df


def fill_nzp_missing_years(nzp_df):
    """
    Fill in missing years in the Net Zero Pathway data by linear interpolation between 5-year gaps.
    Returns a complete NZP dataframe with all years from min to max year.
    """
    if len(nzp_df) == 0:
        return nzp_df

    # Get the year range
    min_year = int(nzp_df['year'].min())
    max_year = int(nzp_df['year'].max())

    # Create complete year range
    all_years = list(range(min_year, max_year + 1))

    # Create new dataframe with all years
    nzp_filled = pd.DataFrame({'year': all_years})

    # Merge with existing data to get the available values
    nzp_filled = nzp_filled.merge(nzp_df, on='year', how='left')

    # Interpolate missing values
    nzp_filled['NZP_lower'] = nzp_filled['NZP_lower'].interpolate(method='linear')
    nzp_filled['NZP_upper'] = nzp_filled['NZP_upper'].interpolate(method='linear')

    # Forward fill and backward fill any remaining NaNs at edges
    nzp_filled['NZP_lower'] = nzp_filled['NZP_lower'].ffill().bfill()
    nzp_filled['NZP_upper'] = nzp_filled['NZP_upper'].ffill().bfill()

    return nzp_filled


def create_emissions_bar_chart_with_wildfires(df, nzp_df, merged_anomaly_df, fire_emissions, start_year=2005):
    """
    Create bar chart version (V3) of emissions gap with stacked wildfire anomaly and NZP bars.
    - Grey bars: Historical national emissions
    - Red stacked bars: Wildfire emissions anomaly (on top of grey bars)
    - Green bars: Net Zero Pathway (mean of upper/lower bounds)
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Filter data to start_year onwards
    df_filtered = df[df['year'] >= start_year].copy()
    merged_filtered = merged_anomaly_df[merged_anomaly_df['year'] >= start_year].copy()

    # Merge to ensure same years
    plot_data = df_filtered.merge(merged_filtered[['year', 'emissions_anomaly_3y_average']], on='year', how='left')
    plot_data['emissions_anomaly_3y_average'] = plot_data['emissions_anomaly_3y_average'].fillna(0)

    # fire emissions anomalies
    positive_anomaly = plot_data['emissions_anomaly_3y_average'].clip(lower=0)

    # Create base bars (historical emissions) in grey
    bars_base = ax.bar(plot_data['year'], plot_data['emissions'],
                       color='#7F8C8D', alpha=0.8, width=0.6,
                       label='Industrial Emissions')

    # Add Net Zero Pathway bars (2025-2050) in green with filled-in missing years
    if len(nzp_df) > 0:
        # Fill in missing years with linear interpolation
        nzp_df_filled = fill_nzp_missing_years(nzp_df)
        nzp_filtered = nzp_df_filled[nzp_df_filled['year'] >= 2025].copy()
        nzp_filtered['nzp_mean'] = (nzp_filtered['NZP_lower'] + nzp_filtered['NZP_upper']) / 2

        bars_nzp = ax.bar(nzp_filtered['year'], nzp_filtered['nzp_mean'],
                          color='#27AE60', alpha=0.7, width=0.6,
                          label='Net Zero Pathway')

    # Formatting
    ax.set_xlim(start_year - 0.5, 2050.5)

    # Calculate max height for y-axis using filled NZP data if available
    if len(nzp_df) > 0:
        nzp_df_filled_for_max = fill_nzp_missing_years(nzp_df)
        nzp_max = nzp_df_filled_for_max['NZP_upper'].max()
    else:
        nzp_max = 0

    max_emissions = max(
        (plot_data['emissions'] + positive_anomaly).max(),
        nzp_max
    )
    ax.set_ylim(0, max_emissions * 1.1)

    # Y-axis formatting
    ax.set_ylabel('', fontsize=14,
                  fontproperties=font_props.get('regular') if font_props else None)

    # X-axis formatting
    ax.set_xticks(range(start_year, 2051, 5))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Create stacked bars (wildfire anomaly) in red on top of base bars
    # Only show positive anomalies as stacked bars
    if fire_emissions:
        bars_anomaly = ax.bar(plot_data['year'], positive_anomaly,
            bottom=plot_data['emissions'],
            color='#E74C3C', alpha=0.8, width=0.6,
            label='Anomalous Wildfire Emissions (5-year mean)')
        
        ax.text(2020, max_emissions, 'ANOMALOUS WILDFIRE\nEMISSIONS',
        fontsize=12, ha='center', va='center',
        fontweight='bold', alpha=0.7, color='#E74C3C')

    # Add legend
    # ax.legend(loc='upper right', fontsize=11)

    # Add text annotations for clarity
    ax.text(2008, max_emissions * .9, 'INDUSTRIAL\nEMISSIONS',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.7, color='#7F8C8D')

    ax.text(2037, max_emissions * 0.4, 'NET ZERO\nPATHWAY',
            fontsize=12, ha='center', va='center',
            fontweight='bold', alpha=0.7, color='#27AE60')

    return fig


def create_wildfire_emissions_bar_chart(wildfire_df):
    """
    Create bar chart of Canada's wildfire emissions 2003-2025.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Filter data to 2003-2025 range
    data_filtered = wildfire_df[
        (wildfire_df['year'] >= 2003) & (wildfire_df['year'] <= 2025)
    ].copy()

    # Create bar chart
    bars = ax.bar(data_filtered['year'], data_filtered['wildfire_emissions'],
                  color='#E67E22', alpha=0.8, width=0.7)

    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_ylabel('Wildfire Emissions (Mt CO₂)', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_xlim(2002.5, 2025.5)
    ax.set_ylim(0, data_filtered['wildfire_emissions'].max() * 1.1)

    # Set x-axis ticks
    ax.set_xticks(range(2003, 2026, 2))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    return fig


def create_wildfire_emissions_line_chart(wildfire_df, country_name='Canada'):
    """
    Create a line chart showing wildfire emissions over time.
    Similar to the Canadian forest carbon balance plot style.

    Parameters:
    -----------
    wildfire_df : pandas.DataFrame
        DataFrame with columns: year, wildfire_emissions
    country_name : str
        Name of the country for labeling

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    from utils import COLORS

    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Plot wildfire emissions as a line - using primary red color for emphasis
    ax.plot(wildfire_df['year'], wildfire_df['wildfire_emissions'],
            color=COLORS['primary'], linewidth=3,
            marker='', solid_capstyle='round', zorder=3)

    # Format y-axis
    ax.tick_params(axis='y', labelsize=14, length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.set_ylim(0, wildfire_df['wildfire_emissions'].max() * 1.1)

    # Format x-axis
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=14, length=0)
    ax.set_xlim(wildfire_df['year'].min() - 1, wildfire_df['year'].max() + 1)

    # Update axis spine colors
    ax.spines['left'].set_color(COLORS['primary'])
    ax.spines['left'].set_linewidth(2)

    return fig


def create_wildfire_emissions_anomaly_bar_chart(wildfire_anomaly_df):
    """
    Create bar chart of Canada's wildfire emissions anomaly with red/blue coloring.
    Positive values (red) above y=0, negative values (blue) below y=0.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Filter data to 2003-2025 range
    data_filtered = wildfire_anomaly_df[
        (wildfire_anomaly_df['year'] >= 2003) & (wildfire_anomaly_df['year'] <= 2025)
    ].copy()

    # Create color array based on positive/negative values
    colors = ['#E74C3C' if val >= 0 else '#3498DB' for val in data_filtered['emissions_anomaly']]

    # Create bar chart
    bars = ax.bar(data_filtered['year'], data_filtered['emissions_anomaly'],
                  color=colors, alpha=0.8, width=0.7)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_ylabel('Wildfire Emissions Anomaly (Mt CO₂)', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_xlim(2002.5, 2025.5)

    # Set y-axis limits with some padding
    max_abs = max(abs(data_filtered['emissions_anomaly'].min()), abs(data_filtered['emissions_anomaly'].max()))
    ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    # Set x-axis ticks
    ax.set_xticks(range(2003, 2026, 2))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    return fig


def create_wildfire_emissions_anomaly_rolling_bar_chart(wildfire_rolling_df):
    """
    Create bar chart of Canada's rolling average wildfire emissions anomaly with red/blue coloring.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Filter data to 2003-2025 range
    data_filtered = wildfire_rolling_df[
        (wildfire_rolling_df['year'] >= 2003) & (wildfire_rolling_df['year'] <= 2025)
    ].copy()

    # Create color array based on positive/negative values
    colors = ['#E74C3C' if val >= 0 else '#3498DB' for val in data_filtered['emissions_anomaly_3y_average']]

    # Create bar chart
    bars = ax.bar(data_filtered['year'], data_filtered['emissions_anomaly_3y_average'],
                  color=colors, alpha=0.8, width=0.7)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_ylabel('Rolling Avg Wildfire Emissions Anomaly (Mt CO₂)', fontsize=14, fontproperties=font_props.get('regular') if font_props else None)
    ax.set_xlim(2002.5, 2025.5)

    # Set y-axis limits with some padding
    max_abs = max(abs(data_filtered['emissions_anomaly_3y_average'].min()), abs(data_filtered['emissions_anomaly_3y_average'].max()))
    ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    # Set x-axis ticks
    ax.set_xticks(range(2003, 2026, 2))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    return fig


def main():
    emissions_file_path = 'data/canada_emissions_gap/canada_emissions_440Mt.xlsx'
    cams_data_dir = 'data/wildfire_emissions'
    canada_shapefile_path = 'data/canada_emissions_gap/ca_shp/ca.shp'

    try:
        # Load and process emissions data
        emissions_df, nzp_df = load_emissions_data(emissions_file_path)
        print(f"Loaded emissions data from {len(emissions_df)} years")
        print(f"Loaded NZP pathway data from {len(nzp_df)} years")

        # Load all CAMS wildfire emissions data (2005-2024)
        wildfire_df = load_all_cams_wildfire_emissions(cams_data_dir, canada_shapefile_path)
        print(f"Loaded CAMS wildfire emissions data from {len(wildfire_df)} years")

        # Merge emissions with wildfire data
        merged_emissions_df = merge_emissions_with_wildfire(emissions_df, wildfire_df)
        print(f"Merged emissions with CAMS wildfire data")

        baseline_avg = calculate_baseline_average(wildfire_df)
        wildfire_anomaly_df = calculate_emissions_anomaly(wildfire_df, baseline_avg)
        wildfire_rolling_df = calculate_rolling_anomaly_average(wildfire_anomaly_df)

        # Merge with emissions data using anomaly adjustments
        merged_anomaly_df = merge_emissions_with_anomaly_adjusted_wildfires(emissions_df, wildfire_rolling_df)
        print(f"Created anomaly-adjusted emissions data")


        # create second version with no fire emissions
        canada_emissions_gap_plot = create_emissions_bar_chart_with_wildfires(emissions_df, nzp_df, merged_anomaly_df, fire_emissions = False, start_year=2005)

        # Calculate emissions gap between 2024 industrial emissions and 2030 pathway
        # Get 2024 industrial emissions
        emissions_2024 = emissions_df[emissions_df['year'] == 2024]['emissions'].values[0]

        # Get 2030 NZP pathway emissions (mean of upper and lower)
        nzp_df_filled = fill_nzp_missing_years(nzp_df)
        nzp_2030 = nzp_df_filled[nzp_df_filled['year'] == 2030]
        if len(nzp_2030) > 0:
            pathway_2030 = (nzp_2030['NZP_lower'].values[0] + nzp_2030['NZP_upper'].values[0]) / 2
        else:
            pathway_2030 = 0

        # Add emissions gap annotation
        if emissions_2024 > pathway_2030:
            annotate_emissions_gap(plt.gca(), 2030, emissions_2024, pathway_2030, units='Mt')

        # Add titles using the utils formatting
        format_plot_title(plt.gca(),
                         "",
                         "Canada's National Emissions (Mt CO\N{SUBSCRIPT TWO}e)",
                         None)

        # Add Deep Sky branding
        add_deep_sky_branding(plt.gca(), None,
                             "DATA: CANADIAN CLIMATE INSTITUTE",
                             analysis_date=datetime.datetime.now())

        # Save the fourth plot
        save_path_bar_2 = 'figures/canada_emissions_gap_bars.png'
        save_plot(canada_emissions_gap_plot, save_path_bar_2)

        print(f"Wildfire V3 bar chart plot saved to {save_path_bar_2}")

        # Create V3 plot (bar chart version)
        canada_emissions_gap_wildfires_plot = create_emissions_bar_chart_with_wildfires(emissions_df, nzp_df, merged_anomaly_df, fire_emissions = True, start_year=2005)

        # Calculate emissions gap between 2025 actual and 2030 pathway
        # Get 2025 industrial emissions
        emissions_2025 = emissions_df[emissions_df['year'] == 2025]['emissions'].values
        if len(emissions_2025) > 0:
            emissions_2025 = emissions_2025[0]
        else:
            # Use 2024 emissions if 2025 not available
            emissions_2025 = emissions_df[emissions_df['year'] == 2024]['emissions'].values[0]

        # Get 2025 wildfire anomaly
        wildfire_anomaly_2025 = merged_anomaly_df[merged_anomaly_df['year'] == 2025]['emissions_anomaly_3y_average'].values
        if len(wildfire_anomaly_2025) > 0:
            wildfire_anomaly_2025 = max(0, wildfire_anomaly_2025[0])  # Only positive anomalies
        else:
            wildfire_anomaly_2025 = 0

        # Total 2025 emissions (industrial + wildfire anomaly)
        total_2025_emissions = emissions_2025 + wildfire_anomaly_2025

        # Get 2030 NZP pathway emissions (mean of upper and lower)
        nzp_df_filled = fill_nzp_missing_years(nzp_df)
        nzp_2030 = nzp_df_filled[nzp_df_filled['year'] == 2030]
        if len(nzp_2030) > 0:
            pathway_2030 = (nzp_2030['NZP_lower'].values[0] + nzp_2030['NZP_upper'].values[0]) / 2
        else:
            pathway_2030 = 0

        # Add emissions gap annotation
        if total_2025_emissions > pathway_2030:
            annotate_emissions_gap(plt.gca(), 2030, total_2025_emissions, pathway_2030, units='Mt')

        # Add titles using the utils formatting
        format_plot_title(plt.gca(),
                         "",
                         "Canada's National Emissions (Mt CO\N{SUBSCRIPT TWO}e)",
                         None)

        # Add Deep Sky branding
        add_deep_sky_branding(plt.gca(), None,
                             "DATA: CANADIAN CLIMATE INSTITUTE, COPERNICUS ATMOSPHERE MONITORING SERVICE: GFAS\n" \
                             "Anomalous wildfire emissions: 10-year rolling average of all wildfire CO\N{SUBSCRIPT TWO} emissions minus 2003-2018 baseline.",
                             analysis_date=datetime.datetime.now())

        # Save the fourth plot
        save_path_bar = 'figures/canada_emissions_gap_bars_wildfires.png'
        save_plot(canada_emissions_gap_wildfires_plot, save_path_bar)

        print(f"Wildfire V3 bar chart plot saved to {save_path_bar}")

        # Create Canada wildfire emissions line chart
        print("\n" + "="*60)
        print("Creating Canada wildfire emissions line chart...")
        print("="*60)

        fig_canada_wildfire = create_wildfire_emissions_line_chart(wildfire_df, country_name='Canada')

        format_plot_title(plt.gca(),
                         "",
                         "CANADA WILDFIRE EMISSIONS",
                         None)

        add_deep_sky_branding(plt.gca(), None,
                             "DATA: COPERNICUS ATMOSPHERE MONITORING SERVICE: GFAS",
                             analysis_date=datetime.datetime.now())

        save_path_canada_wildfire = 'figures/canada_wildfire_emissions.png'
        save_plot(fig_canada_wildfire, save_path_canada_wildfire)

        print(f"Canada wildfire emissions line chart saved to {save_path_canada_wildfire}")

        # Load and create Brazil wildfire emissions line chart
        print("\n" + "="*60)
        print("Loading Brazil wildfire emissions data...")
        print("="*60)

        brazil_shapefile_path = 'data/canada_emissions_gap/ca_shp/br.shp'
        wildfire_brazil_df = load_brazil_wildfire_emissions(cams_data_dir, brazil_shapefile_path)
        print(f"Loaded Brazil CAMS wildfire emissions data from {len(wildfire_brazil_df)} years")
        print(f"Brazil wildfire emissions range: {wildfire_brazil_df['wildfire_emissions'].min():.2f} - {wildfire_brazil_df['wildfire_emissions'].max():.2f} Mt CO2")

        print("\nCreating Brazil wildfire emissions line chart...")
        fig_brazil_wildfire = create_wildfire_emissions_line_chart(wildfire_brazil_df, country_name='Brazil')

        format_plot_title(plt.gca(),
                         "",
                         "BRAZIL WILDFIRE EMISSIONS",
                         None)

        add_deep_sky_branding(plt.gca(), None,
                             "DATA: COPERNICUS ATMOSPHERE MONITORING SERVICE: GFAS",
                             analysis_date=datetime.datetime.now())

        save_path_brazil_wildfire = 'figures/brazil_wildfire_emissions.png'
        save_plot(fig_brazil_wildfire, save_path_brazil_wildfire)

        print(f"Brazil wildfire emissions line chart saved to {save_path_brazil_wildfire}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




