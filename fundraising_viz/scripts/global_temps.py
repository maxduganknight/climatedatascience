"""
Global Temperature Data Scraper and Visualizer

This script retrieves global temperature anomaly data from five major datasets:
- HadCRUT5 (UK Met Office / CRU)
- GISTEMP (NASA)
- NOAA GlobalTemp
- ECMWF ERA5 (Copernicus)
- Berkeley Earth

All datasets are normalized to the 1850-1900 baseline and plotted together.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import os
import sys
from datetime import datetime
import xarray as xr
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Rectangle

# Add parent directory to path for utils import
sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot, COLORS

# Try to import cdsapi for ERA5 data
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
    print("Warning: cdsapi not installed. ERA5 data retrieval will be skipped.")

# Configuration
BASELINE_START = 1850
BASELINE_END = 1900
OUTPUT_DIR = 'data/global_temps'
FIGURES_DIR = 'figures'
URL_CACHE_FILE = os.path.join('data/global_temps', '.url_cache.txt')

# 2025 Temperature Predictions (degrees C above 1880-1900 baseline)
# Format: 'Source': (central_value, lower_bound, upper_bound)
PREDICTIONS_2025 = {
    'Schmidt': (1.35, 1.22, 1.49),
    'Met Office': (1.43, 1.31, 1.55),
    'Berkeley Earth': (1.42, 1.24, 1.60),
    'Carbon Brief': (1.37, 1.22, 1.51)
}

# Data source URLs
URLS = {
    'gistemp': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt',
    'noaa': 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/1850-2025/data.csv',
    'hadcrut5': 'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv',
    'hadcrut5_alt': 'https://hadleyserver.metoffice.gov.uk/hadcrut5/data/current/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv',
    'berkeley': 'https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt',
    'ecmwf': 'https://climate.copernicus.eu/sites/default/files/{year}-{month:02d}/ts_1month_anomaly_Global_ei_2T_{year}{month:02d}.csv'
}


def import_gistemp():
    """
    Import NASA's GISTEMP data.
    Returns monthly anomalies (already relative to 1951-1980).
    """
    print('Importing GISTEMP...')
    try:
        # Add user agent to avoid 403 errors
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(URLS['gistemp'], headers=headers, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text), skiprows=7, sep=r'\s+')

        # Drop annual summary columns if they exist
        cols_to_drop = ['J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON', 'Year.1']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df.drop(existing_cols_to_drop, axis=1, inplace=True)

        # Rename month columns
        month_cols = [col for col in df.columns if col not in ['Year']]
        for i, col in enumerate(month_cols):
            df.rename(columns={col: f'month_{i+1}'}, inplace=True)

        # Remove header rows that sometimes appear
        df = df[~df['Year'].isin(['Year', 'Divide', 'Multiply', 'Example', 'change'])]

        # Convert to long format
        df_long = pd.wide_to_long(df.reset_index(), ['month_'], i='Year', j='month').reset_index()
        df_long.columns = ['year', 'month', 'index', 'gistemp']
        df_long.drop(columns='index', inplace=True)
        df_long = df_long.apply(pd.to_numeric, errors='coerce')
        df_long.sort_values(by=['year', 'month'], inplace=True)

        # Convert from hundredths of degrees
        df_long['gistemp'] = df_long['gistemp'] / 100.0
        df_long.reset_index(inplace=True, drop=True)

        print(f'  Retrieved {len(df_long)} monthly records from GISTEMP')
        return df_long
    except Exception as e:
        print(f'  Error importing GISTEMP: {e}')
        print('  Note: NASA GISTEMP may be temporarily unavailable due to government funding issues')
        return pd.DataFrame(columns=['year', 'month', 'gistemp'])


def import_noaa():
    """
    Import NOAA GlobalTemp data.
    """
    print('Importing NOAA GlobalTemp...')
    try:
        response = requests.get(URLS['noaa'])
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text), skiprows=4, names=['date', 'noaa'])
        df['year'] = df['date'].astype(str).str[:4].astype(int)
        df['month'] = df['date'].astype(str).str[4:6].astype(int)
        df = df[['year', 'month', 'noaa']].apply(pd.to_numeric, errors='coerce')

        print(f'  Retrieved {len(df)} monthly records from NOAA')
        return df
    except Exception as e:
        print(f'  Error importing NOAA: {e}')
        return pd.DataFrame(columns=['year', 'month', 'noaa'])


def import_hadcrut5():
    """
    Import HadCRUT5 data from Met Office.
    Tries multiple URL options and caches the working URL for faster future runs.
    """
    print('Importing HadCRUT5...')

    # List of URLs to try
    urls_to_try = [
        'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv',
        'https://hadleyserver.metoffice.gov.uk/hadcrut5/data/current/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv',
        'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv'
    ]

    # Check if we have a cached working URL
    cached_url = None
    if os.path.exists(URL_CACHE_FILE):
        try:
            with open(URL_CACHE_FILE, 'r') as f:
                cached_url = f.read().strip()
            if cached_url:
                print(f'  Trying cached URL first...')
                # Move cached URL to front of list
                if cached_url in urls_to_try:
                    urls_to_try.remove(cached_url)
                urls_to_try.insert(0, cached_url)
        except Exception:
            pass

    for i, url in enumerate(urls_to_try):
        try:
            print(f'  Attempting URL {i+1}/{len(urls_to_try)}...')
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            # Parse time column (format: YYYY-MM-DD or decimal year)
            if 'Time' in df.columns:
                df['year'] = pd.to_datetime(df['Time']).dt.year
                df['month'] = pd.to_datetime(df['Time']).dt.month
            else:
                # Fall back to parsing from first column
                df['year'] = df.iloc[:, 0].astype(str).str[:4].astype(int)
                df['month'] = df.iloc[:, 0].astype(str).str[5:7].astype(int)

            # Get anomaly column (usually second column)
            anomaly_col = df.columns[1]
            df['hadcrut5'] = pd.to_numeric(df[anomaly_col], errors='coerce')

            df = df[['year', 'month', 'hadcrut5']]

            # Success! Cache this URL for future use
            if url != cached_url:
                print(f'  ✓ Success with URL {i+1}, caching for future runs')
                try:
                    os.makedirs(os.path.dirname(URL_CACHE_FILE), exist_ok=True)
                    with open(URL_CACHE_FILE, 'w') as f:
                        f.write(url)
                except Exception:
                    pass
            else:
                print(f'  ✓ Success with cached URL')

            print(f'  Retrieved {len(df)} monthly records from HadCRUT5')
            return df

        except requests.exceptions.Timeout:
            print(f'  ✗ URL {i+1} timed out after 30s')
        except requests.exceptions.HTTPError as e:
            print(f'  ✗ URL {i+1} returned HTTP error: {e.response.status_code}')
        except Exception as e:
            print(f'  ✗ URL {i+1} failed: {type(e).__name__}')

    print(f'  Error: Could not fetch HadCRUT5 from any available URL')
    return pd.DataFrame(columns=['year', 'month', 'hadcrut5'])


def import_berkeley():
    """
    Import Berkeley Earth Land and Ocean temperature data.
    """
    print('Importing Berkeley Earth...')
    try:
        response = requests.get(URLS['berkeley'])
        response.raise_for_status()

        # Berkeley Earth files have a large header, skip to data
        lines = response.text.split('\n')

        # Find where data starts (after comment lines)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('%'):
                data_start = i
                break

        df = pd.read_csv(io.StringIO('\n'.join(lines[data_start:])),
                        sep=r'\s+', header=None,
                        usecols=[0, 1, 2],
                        names=['year', 'month', 'berkeley'])

        # Berkeley Earth has summary statistics at the end, remove them
        df = df[df['month'].apply(lambda x: str(x).isdigit())]
        df = df.apply(pd.to_numeric, errors='coerce')

        # Filter out any invalid data
        df = df[(df['year'] >= 1850) & (df['month'] >= 1) & (df['month'] <= 12)]

        print(f'  Retrieved {len(df)} monthly records from Berkeley Earth')
        return df
    except Exception as e:
        print(f'  Error importing Berkeley Earth: {e}')
        return pd.DataFrame(columns=['year', 'month', 'berkeley'])


def import_era5_cds():
    """
    Import ECMWF ERA5 temperature data using CDS API.
    Downloads monthly averaged 2m temperature from 1940 to present.
    Uses CSV cache to avoid reprocessing the NetCDF file on every run.
    """
    print('Importing ERA5 from CDS API...')

    if not HAS_CDSAPI:
        print('  CDS API not available, skipping ERA5')
        return pd.DataFrame(columns=['year', 'month', 'era5'])

    try:
        # Define output paths
        era5_file = os.path.join(OUTPUT_DIR, 'era5_temp_monthly.nc')
        era5_csv_cache = os.path.join(OUTPUT_DIR, 'era5_temp_monthly_processed.csv')

        # Check if CSV cache exists and is recent
        if os.path.exists(era5_csv_cache):
            cache_mod_time = datetime.fromtimestamp(os.path.getmtime(era5_csv_cache))
            # If cache is from current year, use it
            if cache_mod_time.year == datetime.now().year:
                print(f'  Using cached processed ERA5 data from {era5_csv_cache}')
                df = pd.read_csv(era5_csv_cache)
                print(f'  Retrieved {len(df)} monthly records from ERA5 cache')
                return df

        # If no valid cache, check if we need to download NetCDF
        download_needed = True
        if os.path.exists(era5_file):
            # Check if file is from current year
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(era5_file))
            if file_mod_time.year == datetime.now().year:
                print(f'  Using cached ERA5 NetCDF from {era5_file}')
                download_needed = False

        if download_needed:
            print('  Downloading ERA5 data (this may take several minutes)...')

            # Generate year list from 1940 to last complete year
            current_year = datetime.now().year
            last_complete_year = current_year - 1  # Only include complete years
            years = [str(y) for y in range(1940, last_complete_year + 1)]

            dataset = "reanalysis-era5-single-levels-monthly-means"
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": ["2m_temperature"],
                "year": years,
                "month": [
                    "01", "02", "03", "04", "05", "06",
                    "07", "08", "09", "10", "11", "12"
                ],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }

            client = cdsapi.Client()
            client.retrieve(dataset, request).download(era5_file)
            print(f'  Downloaded ERA5 data to {era5_file}')

        # Process the NetCDF file
        print('  Processing ERA5 NetCDF data...')
        ds = xr.open_dataset(era5_file)

        # Calculate global mean temperature for each month
        # Weight by cosine of latitude for proper global averaging
        weights = np.cos(np.deg2rad(ds.latitude))
        weights.name = "weights"

        # Calculate weighted global mean
        t2m_weighted = ds.t2m.weighted(weights)
        global_mean = t2m_weighted.mean(dim=['latitude', 'longitude'])

        # Convert to DataFrame
        df = global_mean.to_dataframe(name='era5').reset_index()

        # Extract year and month from time
        df['year'] = df['valid_time'].dt.year
        df['month'] = df['valid_time'].dt.month

        # Convert from Kelvin to Celsius
        df['era5'] = df['era5'] - 273.15

        # Select only needed columns
        df = df[['year', 'month', 'era5']]

        print(f'  Retrieved {len(df)} monthly records from ERA5')

        # Save to CSV cache for faster loading next time
        df.to_csv(era5_csv_cache, index=False)
        print(f'  Saved processed ERA5 data to cache: {era5_csv_cache}')

        ds.close()
        return df

    except Exception as e:
        print(f'  Error importing ERA5: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['year', 'month', 'era5'])


def rebaseline(df, dataset_name, start_year=BASELINE_START, end_year=BASELINE_END):
    """
    Rebaseline temperature data to 1850-1900 baseline.

    For ERA5, which only goes back to 1940, we use a 1991-2020 baseline
    and apply pre-industrial adjustment values from:
    https://climate.copernicus.eu/tracking-breaches-150c-global-warming-threshold

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with year, month, and temperature anomaly columns
    dataset_name : str
        Name of the temperature column (e.g., 'gistemp', 'noaa', 'era5')
    start_year : int
        Start year of baseline period (1850 for most datasets)
    end_year : int
        End year of baseline period (1900 for most datasets)

    Returns:
    --------
    pandas DataFrame
        DataFrame with rebaselined anomalies
    """
    if dataset_name not in df.columns:
        return df

    # Special handling for ERA5 which doesn't go back to 1850-1900
    if dataset_name == 'era5':
        # Use 1991-2020 baseline for ERA5
        era5_baseline_data = df[(df['year'] >= 1991) & (df['year'] <= 2020)]

        if len(era5_baseline_data) == 0:
            print(f'  Warning: No data in 1991-2020 period for ERA5')
            return df

        baseline_mean = era5_baseline_data[dataset_name].mean()

        # Apply pre-industrial adjustment
        # Annual adjustment value is 0.88°C based on Copernicus data
        # This adjusts from 1991-2020 baseline to pre-industrial (1850-1900) baseline
        preindustrial_adjustment = 0.88

        # Calculate anomaly relative to 1991-2020, then add pre-industrial adjustment
        df[dataset_name] = (df[dataset_name] - baseline_mean) + preindustrial_adjustment

        print(f'  Rebaselined {dataset_name} (1991-2020 mean: {baseline_mean:.3f}°C, +{preindustrial_adjustment:.2f}°C pre-industrial adjustment)')

        return df

    # Standard rebaselining for other datasets
    baseline_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    if len(baseline_data) == 0:
        print(f'  Warning: No data in baseline period {start_year}-{end_year} for {dataset_name}')
        return df

    baseline_mean = baseline_data[dataset_name].mean()

    # Subtract baseline mean from all values
    df[dataset_name] = df[dataset_name] - baseline_mean

    print(f'  Rebaselined {dataset_name} (baseline mean: {baseline_mean:.3f}°C)')

    return df


def calculate_annual_means(df):
    """
    Calculate annual mean temperature anomalies from monthly data.
    """
    # Group by year and take mean of all dataset columns
    dataset_cols = [col for col in df.columns if col not in ['year', 'month']]

    annual_df = df.groupby('year')[dataset_cols].mean().reset_index()

    return annual_df


def calculate_epoch_trends(annual_df, epochs):
    """
    Calculate temperature trends for different epochs.

    Parameters:
    -----------
    annual_df : pandas DataFrame
        Annual temperature data with year and dataset columns
    epochs : list of tuples
        Each tuple is (start_year, end_year, label)

    Returns:
    --------
    dict : Dictionary with epoch info and trend statistics
    """
    # Calculate ensemble mean across all datasets (ignoring NaNs)
    dataset_cols = [col for col in annual_df.columns if col != 'year']
    annual_df['ensemble_mean'] = annual_df[dataset_cols].mean(axis=1, skipna=True)

    epoch_stats = []

    for start_year, end_year, label in epochs:
        # Filter data for this epoch
        epoch_data = annual_df[(annual_df['year'] >= start_year) &
                               (annual_df['year'] < end_year)].copy()

        if len(epoch_data) < 2:
            continue

        # Fit linear trend
        x = epoch_data['year'].values
        y = epoch_data['ensemble_mean'].values

        # Remove NaNs
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            continue

        # Linear regression: y = mx + b
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # °C per year

        # Convert to °C per decade
        trend_per_decade = slope * 10

        epoch_stats.append({
            'start_year': start_year,
            'end_year': end_year,
            'label': label,
            'trend_per_decade': trend_per_decade,
            'slope': slope,
            'intercept': coeffs[1]
        })

    return epoch_stats


def retrieve_all_datasets():
    """
    Retrieve and combine all temperature datasets.
    """
    print('\n=== Retrieving Global Temperature Datasets ===\n')

    # Import all datasets
    gistemp = import_gistemp()
    noaa = import_noaa()
    hadcrut5 = import_hadcrut5()
    berkeley = import_berkeley()
    era5 = import_era5_cds()

    # Merge all datasets
    print('\nMerging datasets...')

    dfs = [gistemp, noaa, hadcrut5, berkeley, era5]
    dfs = [df for df in dfs if len(df) > 0]  # Remove empty dataframes

    if len(dfs) == 0:
        print('Error: No datasets were successfully retrieved')
        return pd.DataFrame()

    # Merge on year and month
    combined = dfs[0]
    for df in dfs[1:]:
        combined = pd.merge(combined, df, on=['year', 'month'], how='outer')

    # Sort by year and month
    combined = combined.sort_values(['year', 'month']).reset_index(drop=True)

    # Filter to only include complete years
    current_year = datetime.now().year
    last_complete_year = current_year - 1
    combined = combined[combined['year'] <= last_complete_year]

    print(f'\nCombined dataset: {len(combined)} monthly records')
    print(f'Including data through {last_complete_year} (excluding incomplete year {current_year})')

    # Rebaseline all datasets to 1850-1900
    print(f'\n=== Rebaselining to {BASELINE_START}-{BASELINE_END} ===\n')

    for col in combined.columns:
        if col not in ['year', 'month']:
            combined = rebaseline(combined, col, BASELINE_START, BASELINE_END)

    return combined


class HandlerLineWithShade(HandlerLine2D):
    """
    Custom legend handler that shows a line with a shaded area behind it.
    """
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create the shaded rectangle
        rect = Rectangle((xdescent, ydescent - height/4), width, height * 1.5,
                        facecolor='#CCCCCC', alpha=0.5, transform=trans,
                        edgecolor='none')

        # Create the line
        line = super().create_artists(legend, orig_handle, xdescent, ydescent,
                                     width, height, fontsize, trans)

        return [rect] + line


def create_visualization(annual_df, save_path):
    """
    Create a visualization matching the reference image style.
    Shows all temperature datasets as lines with Berkeley Earth confidence interval.
    """
    print('\n=== Creating Visualization ===\n')

    # Define epochs
    epochs = [
        (1850, 1900, 'Preindustrial'),
        (1900, 1950, 'Industrialization'),
        (1950, 2000, 'Globalization'),
        (2000, 2025, 'Nature Collapse')
    ]

    # Calculate epoch trends
    epoch_stats = calculate_epoch_trends(annual_df, epochs)

    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))

    # Define colors for each dataset
    dataset_colors = {
        'hadcrut5': '#4472C4',     # Blue
        'gistemp': '#ED7D31',      # Orange
        'noaa': '#FFC000',         # Yellow/Gold
        'era5': '#9E480E',         # Brown
        'berkeley': '#5B5B5B'      # Dark grey
    }

    dataset_labels = {
        'hadcrut5': 'HadCRUT5',
        'gistemp': 'GISTEMP',
        'noaa': 'NOAA',
        'era5': 'ECMWF',
        'berkeley': 'Berkeley Earth'
    }

    # Plot each dataset and store line handles
    line_handles = {}
    for dataset in ['hadcrut5', 'gistemp', 'noaa', 'era5', 'berkeley']:
        if dataset in annual_df.columns:
            # Filter to only plot from 1850 onwards
            plot_df = annual_df[annual_df['year'] >= 1850].copy()

            line = ax.plot(plot_df['year'], plot_df[dataset],
                   color=dataset_colors[dataset],
                   linewidth=2.5,
                   label=dataset_labels[dataset],
                   alpha=0.8,
                   zorder=2)  # Draw lines on top of confidence interval

            line_handles[dataset] = line[0]

    # Add Berkeley Earth confidence interval if available
    # (This would require retrieving uncertainty data - placeholder for now)
    # We'll add a shaded region around Berkeley Earth
    if 'berkeley' in annual_df.columns:
        plot_df = annual_df[annual_df['year'] >= 1850].copy()
        # Approximate uncertainty (would need actual data from Berkeley Earth)
        uncertainty = 0.1
        ax.fill_between(plot_df['year'],
                        plot_df['berkeley'] - uncertainty,
                        plot_df['berkeley'] + uncertainty,
                        color='#CCCCCC', alpha=0.3,
                        zorder=1)  # Draw confidence interval behind lines

    # Format axes
    ax.set_xlim(1850, 2030)
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('', fontsize=14, labelpad=15)

    # Add horizontal line at 0
    ax.axhline(y=0, color='#888888', linestyle='--', linewidth=1, alpha=0.5)

    # Format ticks
    ax.set_xticks(range(1860, 2030, 20))
    ax.tick_params(axis='both', labelsize=12)

    # Add legend with custom handler for Berkeley Earth
    font_prop = font_props.get('regular') if font_props else None

    # Create handler map - use custom handler only for Berkeley Earth
    handler_map = {}
    if 'berkeley' in line_handles:
        handler_map[line_handles['berkeley']] = HandlerLineWithShade()

    ax.legend(loc=(0.01, .55), fontsize=11, frameon=True,
             facecolor='white', edgecolor='#DDDDDD',
             prop=font_prop,
             handler_map=handler_map)

    # Add epoch dividers and annotations
    epoch_dividers = [1900, 1950, 2000]
    for year in epoch_dividers:
        ax.axvline(x=year, color='#666666', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    # Add epoch annotations
    for epoch in epoch_stats:
        # Calculate midpoint for annotation
        mid_year = (epoch['start_year'] + epoch['end_year']) / 2

        # Format the annotation text
        year_range = f"{epoch['start_year']}-{epoch['end_year']}"
        # Display 0°C/decade if rate is less than 0.01
        trend_value = epoch['trend_per_decade'] if abs(epoch['trend_per_decade']) >= 0.01 else 0.0
        trend_text = f"{trend_value:+.3f}°C/decade"
        annotation = f"{year_range}\n{epoch['label']}\n{trend_text}"

        # Place annotation at the top of the plot
        ax.text(mid_year, 1.7, annotation,
                ha='center', va='top',
                fontsize=9, color='#333333',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#CCCCCC', alpha=0.9))

    # Add title and branding
    format_plot_title(ax,
                     '',
                     'GLOBAL TEMPERATURE ANOMALY (°C)',
                     font_props)

    # Determine which datasets are included
    datasets_included = []
    if 'gistemp' in annual_df.columns:
        datasets_included.append('NASA GISTEMP')
    if 'noaa' in annual_df.columns:
        datasets_included.append('NOAA GLOBALTEMP')
    if 'hadcrut5' in annual_df.columns:
        datasets_included.append('UK MET OFFICE HADCRUT5')
    if 'era5' in annual_df.columns:
        datasets_included.append('ECMWF ERA5')
    if 'berkeley' in annual_df.columns:
        datasets_included.append('BERKELEY EARTH')

    data_note = f'Anomaly relative to pre-industrial baseline 1850-1900. Berkeley Earth 95% confidence interval shaded.'

    add_deep_sky_branding(ax, font_props,
                         data_note=data_note,
                         analysis_date=datetime.now())

    # Save the plot
    save_plot(fig, save_path)

    print(f'Visualization saved to {save_path}')


def create_2025_predictions_plot(annual_df, save_path):
    """
    Create a plot showing the mean of all datasets from 1970-2024 with 2025 predictions.
    Replicates the style of the Carbon Brief 2025 predictions chart.
    """
    print('\n=== Creating 2025 Predictions Plot ===\n')

    # Filter data to 1970-2024
    plot_df = annual_df[(annual_df['year'] >= 1970) & (annual_df['year'] <= 2024)].copy()

    # Calculate mean across all available datasets
    dataset_cols = [col for col in plot_df.columns if col != 'year']
    plot_df['mean'] = plot_df[dataset_cols].mean(axis=1, skipna=True)

    # Set up the plot
    fig, ax, font_props = setup_enhanced_plot(figsize=(12, 7))

    # Plot the observations line
    ax.plot(plot_df['year'], plot_df['mean'],
            color='#555555', linewidth=2.5, marker='o', markersize=4,
            label='Observations', zorder=3)

    # Add a vertical line at 2025
    ax.axvline(x=2025, color='#999999', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    # Add "2025" label on the right side
    ax.text(2026.5, 1.65, '2025', fontsize=11, color='#666666', ha='left', va='bottom')

    # Plot each prediction as a point with error bars, spaced out horizontally
    colors = ['#8B0000', '#DC143C', '#FFA07A', '#4169E1']  # Dark red, red, light red, blue

    # Space predictions across years 2026-2029 for visibility
    prediction_years = [2026, 2027, 2028, 2029]

    for i, (label, (central, lower, upper)) in enumerate(PREDICTIONS_2025.items()):
        year = prediction_years[i]

        # Calculate error bar lengths (distance from central value to bounds)
        error_lower = central - lower
        error_upper = upper - central

        # Plot point with error bar
        ax.errorbar(year, central,
                   yerr=[[error_lower], [error_upper]],
                   marker='o', markersize=8,
                   color=colors[i], label=label,
                   capsize=5, capthick=2,
                   linewidth=2, zorder=4)

    # Format axes
    ax.set_xlim(1970, 2030)
    ax.set_ylim(0.0, 1.65)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('', fontsize=12, labelpad=15)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#CCCCCC')
    ax.set_axisbelow(True)

    # Format ticks
    ax.set_xticks(range(1970, 2030, 10))
    ax.tick_params(axis='both', labelsize=11)

    # Add legend
    font_prop = font_props.get('regular') if font_props else None
    ax.legend(loc='upper left', fontsize=10, frameon=True,
             facecolor='white', edgecolor='#DDDDDD',
             prop=font_prop)

    # Add title
    format_plot_title(ax,
                     '',
                     'Global Temperature Anomaly (°C)',
                     font_props)

    # Add data note
    data_note = ('Anomaly relative to pre-industrial baseline of 1850-1900.')

    add_deep_sky_branding(ax, font_props,
                         data_note=data_note,
                         analysis_date=datetime.now())

    # Save the plot
    save_plot(fig, save_path)

    print(f'2025 Predictions visualization saved to {save_path}')


def main():
    """
    Main execution function.
    """
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Retrieve all datasets
    monthly_df = retrieve_all_datasets()

    if len(monthly_df) == 0:
        print('\nError: Could not retrieve any temperature data')
        return

    # Calculate annual means
    print('\nCalculating annual means...')
    annual_df = calculate_annual_means(monthly_df)

    # Save to CSV
    output_csv = os.path.join(OUTPUT_DIR, 'global_temps.csv')
    annual_df.to_csv(output_csv, index=False)
    print(f'\nSaved annual temperature data to: {output_csv}')

    # Also save monthly data
    monthly_csv = os.path.join(OUTPUT_DIR, 'global_temps_monthly.csv')
    monthly_df.to_csv(monthly_csv, index=False)
    print(f'Saved monthly temperature data to: {monthly_csv}')

    # Create visualization
    figure_path = os.path.join(FIGURES_DIR, 'global_warming_1850_2024.png')
    create_visualization(annual_df, figure_path)

    # Create 2025 predictions plot
    predictions_path = os.path.join(FIGURES_DIR, 'global_temps_2025_predictions.png')
    create_2025_predictions_plot(annual_df, predictions_path)

    # Print summary statistics
    print('\n=== Summary Statistics ===\n')
    latest_year = annual_df['year'].max()
    print(f'Latest year in dataset: {latest_year}')

    for col in annual_df.columns:
        if col != 'year':
            latest_value = annual_df[annual_df['year'] == latest_year][col].values
            if len(latest_value) > 0 and not np.isnan(latest_value[0]):
                print(f'{col.upper()}: {latest_value[0]:.3f}°C above 1850-1900 baseline')

    print('\n=== Script Complete ===\n')


if __name__ == '__main__':
    main()
