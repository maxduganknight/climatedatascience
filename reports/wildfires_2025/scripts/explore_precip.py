import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from typing import Optional, Tuple, List, Dict, Any 
import datetime
import geopandas as gpd
import zipfile
import tempfile
import shutil
import matplotlib.dates as mdates
import traceback 

from explore_fwi import (
    find_nc_files, 
    load_nc_fwi_files, 
    get_coord_names, 
    check_coordinate_system, 
    transform_longitude, 
    create_bbox_mask, 
    calculate_fwi_anomaly
)

from utils import (
    setup_space_mono_font, setup_enhanced_plot, format_plot_title,
    add_deep_sky_branding, add_legend, save_plot
)

# --- Constants ---
PRECIP_VAR_NAME = 'tp' # Example variable name for precipitation
FWI_VAR_NAME = 'fwinx' # Example variable name for FWI

# --- Helper Functions ---

def _find_time_coord_name(data: xr.Dataset) -> str:
    """Find the name of the time coordinate in the dataset."""
    possible_time_coords = ['time', 'valid_time', 't']
    for coord in possible_time_coords:
        if coord in data.coords:
            return coord
    raise ValueError("Could not determine the time coordinate in the dataset.")

def load_monthly_precip_from_zips(zip_file_paths: List[str]) -> Optional[xr.Dataset]:
    """
    Load and combine monthly precipitation data from ZIP files containing NetCDF files.

    Assumes each ZIP contains one relevant NetCDF file and the data inside is monthly.
    """
    print(f"\nProcessing {len(zip_file_paths)} precipitation ZIP files...")
    valid_datasets = []
    problematic_files = []
    temp_dirs = [] # Keep track of temp dirs to ensure cleanup

    try:
        for zip_file_path in zip_file_paths:
            print(f"Processing: {os.path.basename(zip_file_path)}")
            temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
            nc_file_opened = False
            ds = None # Initialize ds to None
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                nc_files = [f for f in os.listdir(temp_dir) if f.endswith('.nc')]
                if not nc_files:
                    raise FileNotFoundError("No NetCDF file found within ZIP.")

                extracted_nc_path = os.path.join(temp_dir, nc_files[0])
                print(f"  Found and opening NetCDF: {nc_files[0]}")
                ds = xr.open_dataset(extracted_nc_path, engine='netcdf4')
                nc_file_opened = True
                # Basic check for monthly frequency (optional but recommended)
                time_coord_name = _find_time_coord_name(ds)
                time_diff = np.diff(ds[time_coord_name].values)
                if not np.all((time_diff / np.timedelta64(1, 'D')) > 25):
                     warnings.warn(f"Data in {nc_files[0]} might not be monthly based on time steps.", UserWarning)
                
                valid_datasets.append(ds)
                print(f"  Successfully opened: {nc_files[0]}")

            except (zipfile.BadZipFile, FileNotFoundError, Exception) as e:
                error_type = type(e).__name__
                print(f"  ERROR processing {os.path.basename(zip_file_path)}: {error_type} - {e}")
                problematic_files.append(f"{zip_file_path} ({error_type})")
                if nc_file_opened and ds:
                    ds.close() # Close if opened before error

    finally:
        # Ensure all temporary directories are removed
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    if problematic_files:
        print("\n-------------------------------------")
        print("Summary of problematic files:")
        for f in problematic_files: print(f" - {f}")
        print("-------------------------------------")

    if not valid_datasets:
        print("\nError: No valid precipitation datasets could be loaded.")
        return None

    print(f"\nCombining {len(valid_datasets)} valid precipitation datasets...")
    try:
        # Combine datasets (ensure time coordinate is consistent)
        combined_data = xr.combine_by_coords(valid_datasets, combine_attrs="override")
        # Close individual datasets after combining
        for ds in valid_datasets:
            ds.close()

        time_coord_name = _find_time_coord_name(combined_data)
        print("\nCombined Precipitation Dataset dimensions:")
        print(f"Dimensions: {combined_data.dims}")
        print(f"Time range: {pd.to_datetime(combined_data[time_coord_name].min().item()).strftime('%Y-%m-%d')} to {pd.to_datetime(combined_data[time_coord_name].max().item()).strftime('%Y-%m-%d')}")
        return combined_data

    except Exception as e:
        print(f"\nError combining precipitation datasets: {e}")
        # Ensure files are closed if combination fails
        for ds in valid_datasets:
            try: ds.close()
            except: pass
        return None

# --- Anomaly Calculation ---

def _calculate_period_value(data_period: xr.Dataset, data_var: str, aggregation_method: str, time_coord_name: str) -> xr.DataArray:
    """Helper to calculate mean or sum over the time dimension."""
    if aggregation_method == 'mean':
        return data_period[data_var].mean(dim=time_coord_name, skipna=True)
    elif aggregation_method == 'sum':
        return data_period[data_var].sum(dim=time_coord_name, skipna=True)
    else:
        raise ValueError("aggregation_method must be 'mean' or 'sum'")

def calculate_annual_regional_sum(data: xr.Dataset, data_var: str, region_mask: xr.DataArray, lat_name: str, lon_name: str) -> Optional[pd.Series]:
    """
    Calculates the spatially averaged annual sum for a variable within a region.
    Assumes monthly data for precipitation (sums months) or daily data (sums days).
    """
    try:
        time_coord_name = _find_time_coord_name(data)
        print(f"  Calculating annual regional sum for {data_var}...")

        # Apply region mask first
        regional_data = data[data_var].where(region_mask)

        # Group by year and sum over time (all months/days within the year)
        annual_sum_spatial = regional_data.groupby(f"{time_coord_name}.year").sum(dim=time_coord_name, skipna=True)

        # Then calculate spatial mean over the region
        annual_regional_mean_of_sum = annual_sum_spatial.sum(dim=[lat_name, lon_name], skipna=True) # Renamed for clarity

        # Compute the result and convert to pandas Series
        annual_regional_mean_of_sum_computed = annual_regional_mean_of_sum.compute() # Renamed for clarity

        if annual_regional_mean_of_sum_computed.size == 0:
             print(f"  Error: No annual sums could be calculated for {data_var}.")
             return None

        print(f"  Finished calculating annual regional sum for {data_var}.")
        return annual_regional_mean_of_sum_computed.to_series()

    except Exception as e:
        print(f"Error calculating annual regional sum for {data_var}: {e}")
        traceback.print_exc()
        return None

def calculate_regional_fwi_percentile(data: xr.Dataset, data_var: str, baseline_years: range, region_mask: xr.DataArray, percentile: float = 95.0) -> Optional[float]:
    """
    Calculates a specific percentile of daily FWI values within a region
    over the baseline period.
    """
    try:
        time_coord_name = _find_time_coord_name(data)
        print(f"  Calculating {percentile}th percentile regional FWI for baseline {baseline_years.start}-{baseline_years.stop-1}...")
        
        # DEBUG: Print data structure and available variables
        print(f"  DEBUG: FWI data dimensions: {data.dims}")
        print(f"  DEBUG: FWI data coordinates: {list(data.coords.keys())}")
        print(f"  DEBUG: FWI data variables: {list(data.variables)}")
        
        # Check if the variable exists
        if data_var not in data.variables:
            print(f"  ERROR: Variable '{data_var}' not found in FWI data. Available variables: {list(data.variables)}")
            return None
        
        # Check coordinate ranges to ensure compatibility with mask
        print(f"  DEBUG: FWI latitude range: {data['latitude'].min().item()} to {data['latitude'].max().item()}")
        print(f"  DEBUG: FWI longitude range: {data['longitude'].min().item()} to {data['longitude'].max().item()}")
        print(f"  DEBUG: Mask latitude range: {region_mask['latitude'].min().item()} to {region_mask['latitude'].max().item()}")
        print(f"  DEBUG: Mask longitude range: {region_mask['longitude'].min().item()} to {region_mask['longitude'].max().item()}")

        # Select baseline years
        baseline_data = data.sel({time_coord_name: data[time_coord_name].dt.year.isin(baseline_years)})
        if baseline_data[time_coord_name].size == 0:
            print("  Error: No FWI data found in baseline years.")
            return None

        # Apply region mask
        regional_baseline_fwi = baseline_data[data_var].where(region_mask)
        
        # DEBUG: Check if we have valid data after masking
        print(f"  DEBUG: Masked FWI data has {regional_baseline_fwi.size} elements")
        print(f"  DEBUG: Masked FWI data dimensions: {regional_baseline_fwi.dims}")
        print(f"  DEBUG: Number of non-NaN values: {np.sum(~np.isnan(regional_baseline_fwi.values))}")
        
        # # If no valid data points, try to create a new mask specifically for FWI data
        # if np.sum(~np.isnan(regional_baseline_fwi.values)) == 0:
        #     print("  DEBUG: No valid data after masking. Attempting to create a new mask for FWI data...")
            
        #     # Create a new mask for FWI data
        #     from explore_fwi import create_bbox_mask, check_coordinate_system
        #     transform_region_coords_fwi = check_coordinate_system(data, REGION_COORDS['west'], REGION_COORDS['east'])
        #     fwi_mask = create_bbox_mask(
        #         data,
        #         REGION_COORDS['west'], REGION_COORDS['east'],
        #         REGION_COORDS['south'], REGION_COORDS['north'],
        #         transform_region_coords_fwi
        #     )
            
        #     # Apply the new mask
        #     regional_baseline_fwi = baseline_data[data_var].where(fwi_mask)
        #     print(f"  DEBUG: After creating new mask - Non-NaN values: {np.sum(~np.isnan(regional_baseline_fwi.values))}")
            
        #     # If still no valid data, return None
        #     if np.sum(~np.isnan(regional_baseline_fwi.values)) == 0:
        #         print("  DEBUG: Still no valid data after creating a new mask.")
        #         return None
        
        # Try to calculate percentile by explicitly specifying dimensions instead of using xr.ALL_DIMS
        # List all dimensions explicitly
        all_dims = list(regional_baseline_fwi.dims)
        print(f"  DEBUG: Will compute percentile across these dimensions: {all_dims}")
        
        # Calculate the percentile across all valid values using explicit dimensions
        try:
            percentile_value = regional_baseline_fwi.quantile(percentile / 100.0, dim=all_dims, skipna=True).compute().item()
        except TypeError as te:
            print(f"  DEBUG: TypeError in quantile calculation: {te}")
            # Alternative approach: convert to numpy array and calculate percentile
            print("  DEBUG: Trying alternative approach with numpy")
            flattened_data = regional_baseline_fwi.values.flatten()
            valid_data = flattened_data[~np.isnan(flattened_data)]
            if len(valid_data) == 0:
                print("  DEBUG: No valid data points found after filtering NaNs")
                return None
                
            percentile_value = np.percentile(valid_data, percentile)
            print(f"  DEBUG: Numpy percentile calculation result: {percentile_value}")

        if np.isnan(percentile_value):
            print(f"  Error: Could not calculate valid {percentile}th percentile FWI.")
            return None

        print(f"  {percentile}th Percentile Regional FWI: {percentile_value:.4f}")
        return percentile_value

    except Exception as e:
        print(f"Error calculating regional FWI percentile: {e}")
        traceback.print_exc()
        return None

def count_annual_extreme_fwi_days(data: xr.Dataset, data_var: str, region_mask: xr.DataArray, fwi_threshold: float, lat_name: str, lon_name: str) -> Optional[pd.Series]:
    """
    Counts the number of days per year where the regional average FWI exceeds a threshold.
    """
    try:
        time_coord_name = _find_time_coord_name(data)
        print(f"  Counting annual extreme FWI days (threshold > {fwi_threshold:.4f})...")

        # Calculate daily regional average FWI
        daily_regional_mean_fwi = data[data_var].where(region_mask).mean(dim=[lat_name, lon_name], skipna=True)

        # Identify days exceeding the threshold
        extreme_days = daily_regional_mean_fwi > fwi_threshold

        # Group by year and sum the boolean mask (True=1, False=0)
        annual_extreme_count = extreme_days.groupby(f"{time_coord_name}.year").sum(dim=time_coord_name, skipna=False) # Don't skipna here

        # Compute and convert to Series
        annual_extreme_count_computed = annual_extreme_count.compute()
        
        if annual_extreme_count_computed.size == 0:
             print(f"  Error: No annual extreme FWI day counts could be calculated.")
             return None

        print(f"  Finished counting annual extreme FWI days. ")
        print(f"  Annual extreme FWI days count: {annual_extreme_count_computed}")
        return annual_extreme_count_computed.to_series()

    except Exception as e:
        print(f"Error counting annual extreme FWI days: {e}")
        traceback.print_exc()
        return None

def plot_annual_precip_anomalies(
    precip_anomalies: pd.Series,
    region_name: str,
    output_dir: str,
    start_year: int = 1995,
    current_year: int = 2025,
    ytd_end_month: int = 5,  
    ytd_end_day: int = 15,
    use_zscore: bool = True
):
    """
    Plots annual total precipitation anomaly as bars with positive/negative coloring.
    Uses Deep Sky branding and styling.
    
    Parameters:
    -----------
    precip_anomalies : pd.Series
        Series containing annual precipitation anomalies with years as index.
        If use_zscore=True, these should be standardized anomalies (z-scores).
    region_name : str
        Name of the region for the title
    output_dir : str
        Directory to save the output figure
    start_year : int
        First year to include in plot
    current_year : int
        The current year (typically using YTD data)
    ytd_end_month, ytd_end_day : int
        The month and day defining the YTD period for current year
    use_zscore : bool
        Whether the input anomalies are standardized (z-scores)
    """
    if precip_anomalies is None or precip_anomalies.empty:
        print("Skipping annual precipitation plot due to missing data.")
        return

    # Filter for expected year range
    precip_anomalies = precip_anomalies.loc[start_year:current_year]
    years = precip_anomalies.index.values
    
    # Set up the enhanced plot with Deep Sky styling
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))
    
    # Create bars with different colors for positive/negative values
    pos_mask = precip_anomalies >= 0
    neg_mask = precip_anomalies < 0
    
    bar_width = 0.6
    ax.bar(years[pos_mask], precip_anomalies[pos_mask], bar_width, 
           color='tab:blue', alpha=0.7, label='Positive Anomaly')
    ax.bar(years[neg_mask], precip_anomalies[neg_mask], bar_width, 
           color='tab:red', alpha=0.7, label='Negative Anomaly')
    
    # Add horizontal line at zero
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Format y-axis without rotating label
    if use_zscore:
        subtitle = "STANDARDIZED PRECIPITATION ANOMALY"
        title_note = f"Note: {current_year} shows Jan 1-{ytd_end_month}/{ytd_end_day} YTD standardized anomaly (σ units)"
    else:
        subtitle = "ANNUAL PRECIPITATION ANOMALY (MM)"
        title_note = f"Note: {current_year} shows Jan 1-{ytd_end_month}/{ytd_end_day} YTD anomaly"
    
    # Use the Deep Sky title formatting
    format_plot_title(
        ax, 
        f"HYDROCLIMATE WHIPLASH IN {region_name.upper()}", 
        subtitle, 
        font_props
    )
    
    # Format x-axis ticks to avoid crowding
    ax.set_xlabel('')  # No x-axis label needed
    ax.set_xticks(years[::2])  # Label every other year
    ax.tick_params(axis='x', rotation=45)
    
    # Add Deep Sky branding with data note
    add_deep_sky_branding(
        ax, 
        font_props, 
        data_note=f"DATA: ERA5 CLIMATE REANALYSIS. 2025 ANOMALY IS YEAR TO DATE THROUGH {ytd_end_month}/{ytd_end_day}."
    )
    
    # Save figure using the shared save function
    plot_filename = os.path.join(output_dir, f"{region_name.replace(' ', '_').lower()}_annual_precip_anomalies.png")
    save_plot(fig, plot_filename)
    
    plt.close(fig)

def plot_annual_precip_anomalies_french(
    precip_anomalies: pd.Series,
    region_name: str,
    output_dir: str,
    start_year: int = 1995,
    current_year: int = 2025,
    ytd_end_month: int = 5,  
    ytd_end_day: int = 15,
    use_zscore: bool = True
):
    """
    French version of annual precipitation anomaly plot.
    """
    if precip_anomalies is None or precip_anomalies.empty:
        print("Skipping French annual precipitation plot due to missing data.")
        return

    # Filter for expected year range
    precip_anomalies = precip_anomalies.loc[start_year:current_year]
    years = precip_anomalies.index.values
    
    # Set up the enhanced plot with Deep Sky styling
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))
    
    # Create bars with different colors for positive/negative values
    pos_mask = precip_anomalies >= 0
    neg_mask = precip_anomalies < 0
    
    bar_width = 0.6
    ax.bar(years[pos_mask], precip_anomalies[pos_mask], bar_width, 
           color='tab:blue', alpha=0.7, label='Anomalie Positive')
    ax.bar(years[neg_mask], precip_anomalies[neg_mask], bar_width, 
           color='tab:red', alpha=0.7, label='Anomalie Négative')
    
    # Add horizontal line at zero
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # French region name translations
    region_translations = {
        'Southwest US & Northern Mexico': 'Sud-Ouest des États-Unis et Nord du Mexique',
        'US Southwest': 'Sud-Ouest des États-Unis',
        'Southern California': 'Californie du Sud',
        'California': 'Californie'
    }
    
    french_region = region_translations.get(region_name, region_name)
    
    # Format y-axis without rotating label
    if use_zscore:
        subtitle = "ANOMALIE DE PRÉCIPITATION STANDARDISÉE"
        title_note = f"Note: {current_year} montre l'anomalie standardisée cumulative du 1er janvier au {ytd_end_day}/{ytd_end_month} (unités σ)"
    else:
        subtitle = "ANOMALIE DE PRÉCIPITATION ANNUELLE (MM)"
        title_note = f"Note: {current_year} montre l'anomalie cumulative du 1er janvier au {ytd_end_day}/{ytd_end_month}"
    
    # Use the Deep Sky title formatting with French text
    format_plot_title(
        ax, 
        f"FOUET HYDROCLIMATIQUE DANS {french_region.upper()}", 
        subtitle, 
        font_props
    )
    
    # Format x-axis ticks to avoid crowding
    ax.set_xlabel('')  # No x-axis label needed
    ax.set_xticks(years[::2])  # Label every other year
    ax.tick_params(axis='x', rotation=45)
    
    # Add Deep Sky branding with French data note
    add_deep_sky_branding(
        ax, 
        font_props, 
        data_note=f"DONNÉES : ERA5 CLIMATE REANALYSIS. L'anomalie 2025 est cumulative jusqu'au {ytd_end_day}/{ytd_end_month}."
    )
    
    # Save figure using the shared save function
    plot_filename = os.path.join(output_dir, f"{region_name.replace(' ', '_').lower()}_annual_precip_anomalies_fr.png")
    save_plot(fig, plot_filename)
    
    plt.close(fig)

def calculate_ytd_regional_precipitation(
    data: xr.Dataset, 
    data_var: str, 
    region_mask: xr.DataArray, 
    baseline_years: range,
    end_month: int = 5,  # Default to May (1-12)
    end_day: int = 1,    # Default to first day of the month
    lat_name: str = 'latitude', 
    lon_name: str = 'longitude'
) -> Tuple[pd.Series, float]:
    """
    Calculates the regional precipitation sum for Jan 1 to a specified date
    for each year in the dataset, and the baseline average.
    
    Returns:
        Tuple of (year-to-date regional precip sums Series, baseline average)
    """
    try:
        time_coord_name = _find_time_coord_name(data)
        print(f"  Calculating year-to-date (through month {end_month}) regional precipitation...")
        
        # Apply region mask
        regional_data = data[data_var].where(region_mask)
        
        # Prepare to store year-to-date values
        ytd_sums = {}
        
        # Get all available years in the dataset
        years = sorted(set(data[time_coord_name].dt.year.values))
        
        for year in years:
            # Select data for current year up to end_month/end_day
            cutoff_date = np.datetime64(f"{year}-{end_month:02d}-{end_day:02d}")
            year_data = regional_data.sel({
                time_coord_name: (
                    (regional_data[time_coord_name].dt.year == year) & 
                    (regional_data[time_coord_name] <= cutoff_date)
                )
            })
            
            # Skip years with no data
            if year_data.size == 0:
                print(f"    No data available for year {year} through {end_month:02d}/{end_day:02d}")
                continue
                
            # Sum over time and then average over region
            ytd_sum = year_data.sum(dim=time_coord_name, skipna=True).sum(
                dim=[lat_name, lon_name], skipna=True
            ).compute().item()
            
            ytd_sums[year] = ytd_sum
        
        # Convert to series
        ytd_series = pd.Series(ytd_sums)
        
        # Calculate baseline average for the same period
        baseline_values = [v for y, v in ytd_sums.items() if y in baseline_years]
        if not baseline_values:
            raise ValueError(f"No baseline values found for years {baseline_years.start}-{baseline_years.stop-1}")
        
        baseline_mean = np.mean(baseline_values)
        print(f"  YTD Baseline Average (months 1-{end_month}): {baseline_mean:.2f} mm")
        
        return ytd_series, baseline_mean
        
    except Exception as e:
        print(f"Error calculating YTD precipitation: {e}")
        traceback.print_exc()
        return pd.Series(), np.nan

def plot_ytd_precip_anomalies(
    ytd_anomalies: pd.Series,
    region_name: str,
    end_month: int,
    end_day: int,
    output_dir: str
):
    """
    Plots year-to-date precipitation anomalies with special highlighting for 2025.
    """
    if ytd_anomalies is None or ytd_anomalies.empty:
        print("Skipping YTD plot due to missing data.")
        return
    
    years = ytd_anomalies.index.values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars with different colors based on sign and year
    pos_mask = ytd_anomalies >= 0
    neg_mask = ytd_anomalies < 0
    current_year_mask = ytd_anomalies.index == 2025
    
    bar_width = 0.6
    
    # Regular years (non-2025)
    if sum(pos_mask & ~current_year_mask) > 0:
        ax.bar(years[pos_mask & ~current_year_mask], ytd_anomalies[pos_mask & ~current_year_mask], 
               bar_width, color='tab:blue', alpha=0.7, label='Positive Anomaly')
    
    if sum(neg_mask & ~current_year_mask) > 0:
        ax.bar(years[neg_mask & ~current_year_mask], ytd_anomalies[neg_mask & ~current_year_mask], 
               bar_width, color='tab:red', alpha=0.7, label='Negative Anomaly')
    
    # 2025 with special highlight if it exists in the data
    if sum(current_year_mask & pos_mask) > 0:
        ax.bar(years[current_year_mask & pos_mask], ytd_anomalies[current_year_mask & pos_mask],
               bar_width, color='green', alpha=0.9, label='2025 Positive Anomaly')
    
    if sum(current_year_mask & neg_mask) > 0:
        ax.bar(years[current_year_mask & neg_mask], ytd_anomalies[current_year_mask & neg_mask],
               bar_width, color='darkred', alpha=0.9, label='2025 Negative Anomaly')
    
    # Month name for title
    month_name = datetime.date(2025, end_month, 1).strftime('%B')
    
    # Add horizontal line at zero
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Set axis labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Year-to-Date Precipitation Anomaly (mm)')
    ax.set_title(f'Jan 1 - {month_name} {end_day} Precipitation Anomalies for {region_name}')
    
    # Format x-axis ticks
    if len(years) > 20:
        ax.set_xticks(years[::2])  # Label every other year if many years
    else:
        ax.set_xticks(years)
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    
    # Save figure
    plot_filename = os.path.join(
        output_dir, 
        f"{region_name.replace(' ', '_').lower()}_ytd_{month_name.lower()}_precip_anomalies.png"
    )
    
    try:
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nYTD Precipitation anomaly plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nError saving YTD precipitation anomaly plot: {e}")
    
    plt.close(fig)

def calculate_sept_to_sept_precipitation(
    data: xr.Dataset, 
    data_var: str, 
    region_mask: xr.DataArray, 
    baseline_years: range,
    lat_name: str = 'latitude', 
    lon_name: str = 'longitude'
) -> Tuple[pd.Series, float]:
    """
    Calculates regional precipitation sums from September of one year to September of the next year.
    
    Returns:
        Tuple of (Sept-to-Sept regional precip sums Series, baseline average)
    """
    try:
        time_coord_name = _find_time_coord_name(data)
        print(f"  Calculating September-to-September regional precipitation...")
        
        # Apply region mask
        regional_data = data[data_var].where(region_mask)
        
        # Prepare to store September-to-September values
        sept_to_sept_sums = {}
        
        # Get all available years in the dataset
        years = sorted(set(data[time_coord_name].dt.year.values))
        
        # For each starting year (except the last one)
        for start_year in range(min(years), max(years)):
            end_year = start_year + 1
            
            # Skip if we don't have data for both years
            if start_year not in years or end_year not in years:
                continue
                
            # Define start and end dates
            start_date = np.datetime64(f"{start_year}-09-01")
            end_date = np.datetime64(f"{end_year}-09-30")
            
            # Select data between September of start_year and September of end_year
            period_data = regional_data.sel({
                time_coord_name: (
                    (regional_data[time_coord_name] >= start_date) & 
                    (regional_data[time_coord_name] <= end_date)
                )
            })
            
            # Skip periods with no data
            if period_data.size == 0:
                print(f"    No data available for period {start_year}-09 to {end_year}-09")
                continue
                
            # Sum over time and then average over region
            period_sum = period_data.sum(dim=time_coord_name, skipna=True).mean(
                dim=[lat_name, lon_name], skipna=True
            ).compute().item()
            
            # Store with combined year label
            sept_to_sept_sums[f"{start_year}-{end_year}"] = period_sum
        
        # Convert to series with string index
        sept_to_sept_series = pd.Series(sept_to_sept_sums)
        
        # Calculate baseline average for the same period
        baseline_values = []
        for period_label, value in sept_to_sept_sums.items():
            start_year, end_year = map(int, period_label.split('-'))
            if start_year in baseline_years and end_year in baseline_years:
                baseline_values.append(value)
                
        if not baseline_values:
            raise ValueError(f"No baseline values found for years {baseline_years.start}-{baseline_years.stop-1}")
        
        baseline_mean = np.mean(baseline_values)
        print(f"  September-to-September Baseline Average: {baseline_mean:.2f} mm")
        
        return sept_to_sept_series, baseline_mean
        
    except Exception as e:
        print(f"Error calculating September-to-September precipitation: {e}")
        traceback.print_exc()
        return pd.Series(), np.nan

def plot_sept_to_sept_precip_anomalies(
    sept_to_sept_anomalies: pd.Series,
    region_name: str,
    output_dir: str
):
    """
    Plots September-to-September precipitation anomalies with special highlighting for 2024-2025.
    """
    if sept_to_sept_anomalies is None or sept_to_sept_anomalies.empty:
        print("Skipping September-to-September plot due to missing data.")
        return
    
    # Scale the anomalies to make them more readable (convert to mm)
    scaled_anomalies = sept_to_sept_anomalies * 1000  # Scale up by 1000 for better visibility
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create bars with different colors based on sign and period
    pos_mask = scaled_anomalies >= 0
    neg_mask = scaled_anomalies < 0
    current_period_mask = scaled_anomalies.index == "2024-2025"
    
    bar_width = 0.7
    
    # Use correct x positions by creating a range of positions
    x_positions = np.arange(len(scaled_anomalies))
    
    # Regular periods (non-2024-2025)
    if sum(pos_mask & ~current_period_mask) > 0:
        ax.bar(
            x_positions[pos_mask & ~current_period_mask], 
            scaled_anomalies[pos_mask & ~current_period_mask], 
            bar_width, color='tab:blue', alpha=0.7, label='Positive Anomaly'
        )
    
    if sum(neg_mask & ~current_period_mask) > 0:
        ax.bar(
            x_positions[neg_mask & ~current_period_mask], 
            scaled_anomalies[neg_mask & ~current_period_mask], 
            bar_width, color='tab:red', alpha=0.7, label='Negative Anomaly'
        )
    
    # 2024-2025 with special highlight if it exists
    if "2024-2025" in scaled_anomalies.index and scaled_anomalies.loc["2024-2025"] >= 0:
        current_idx = np.where(scaled_anomalies.index == "2024-2025")[0][0]
        ax.bar(
            x_positions[current_idx],
            scaled_anomalies.loc["2024-2025"],
            bar_width, color='green', alpha=0.9, label='2024-2025 Positive Anomaly'
        )
    
    if "2024-2025" in scaled_anomalies.index and scaled_anomalies.loc["2024-2025"] < 0:
        current_idx = np.where(scaled_anomalies.index == "2024-2025")[0][0]
        ax.bar(
            x_positions[current_idx],
            scaled_anomalies.loc["2024-2025"],
            bar_width, color='darkred', alpha=0.9, label='2024-2025 Negative Anomaly'
        )
    
    # Add horizontal line at zero
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Set axis labels and title
    ax.set_xlabel('Water Year (Sep-Sep)')
    ax.set_ylabel('Precipitation Anomaly (mm)')
    ax.set_title(f'September-to-September Precipitation Anomalies for {region_name}')
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{period[:2]}-{period[-2:]}" for period in scaled_anomalies.index], rotation=45)
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    
    # Save figure
    plot_filename = os.path.join(output_dir, f"{region_name.replace(' ', '_').lower()}_sept_to_sept_precip_anomalies.png")
    try:
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nSeptember-to-September precipitation anomaly plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nError saving September-to-September precipitation anomaly plot: {e}")
    
    plt.close(fig)

def plot_monthly_precip_timeseries(
    data: xr.Dataset,
    data_var: str,
    region_mask: xr.DataArray,
    region_name: str,
    output_dir: str,
    lat_name: str = 'latitude',
    lon_name: str = 'longitude'
):
    """
    Plots the raw monthly precipitation values for the region as a time series.
    """
    try:
        time_coord_name = _find_time_coord_name(data)
        print("Calculating monthly regional precipitation values...")
        
        # Apply region mask and calculate spatial mean for each timestep
        regional_mean_precip = data[data_var].where(region_mask).mean(
            dim=[lat_name, lon_name], skipna=True
        ).compute()
        
        # Convert to pandas for easier plotting
        precip_series = regional_mean_precip.to_series()
        
        # Scale the values if they're too small
        if precip_series.max() < 0.1:
            print("  Scaling precipitation values by 1000 for better visibility")
            precip_series = precip_series * 1000
            y_label = "Regional Mean Precipitation (mm)"
        else:
            y_label = "Regional Mean Precipitation"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot the time series
        precip_series.plot(ax=ax, marker='o', linestyle='-', alpha=0.7)
        
        # Add grid lines and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel(y_label)
        ax.set_title(f'Monthly Precipitation for {region_name} ({precip_series.index.min().strftime("%Y-%m")} to {precip_series.index.max().strftime("%Y-%m")})')
        
        # Improved x-axis formatting with more detailed labels
        # Show year and month labels for key points
        ax.xaxis.set_major_locator(mdates.YearLocator())  # One label per year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))  # Minor ticks in January and July
        
        # Identify and label periods with significant precipitation
        # Find the top N precipitation events to label
        top_n = 3
        top_precip_events = precip_series.nlargest(top_n)
        
        for date, value in top_precip_events.items():
            ax.annotate(f'{date.strftime("%Y-%m")}',
                       xy=(date, value),
                       xytext=(0, 10),  # 10 points vertical offset
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Add more year tick labels for better orientation
        for year in range(precip_series.index.min().year, precip_series.index.max().year + 1, 2):
            ax.axvline(x=pd.Timestamp(f"{year}-01-01"), color='lightgray', linestyle='--', alpha=0.5)
        
        # Highlight recent periods
        # Current year (or most recent full year) in light blue
        max_year = precip_series.index.max().year
        if pd.Timestamp(f"{max_year}-01-01") in precip_series.index:
            current_year_data = precip_series[precip_series.index.year == max_year]
            ax.fill_between(current_year_data.index, 0, current_year_data.values, 
                            alpha=0.2, color='blue', label=f'{max_year}')
        
        ax.legend()
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # Save figure
        plot_filename = os.path.join(output_dir, f"{region_name.replace(' ', '_').lower()}_monthly_precip_timeseries.png")
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Monthly precipitation time series plot saved to {plot_filename}")
        
        plt.close(fig)
        
        return precip_series
    
    except Exception as e:
        print(f"Error plotting monthly precipitation time series: {e}")
        traceback.print_exc()
        return None

def main():
    # --- Configuration ---
    BASE_DATA_DIR = '../data/wildfires/'
    PRECIP_SUBDIR = 'precipitation'
    OUTPUT_DIR = 'figures/precipitation'
    PRECIP_FILE_PATTERN = 'precipitation_north_america*.zip'
    FWI_FILE_PATTERN = 'CEMS_fwi_north_america_*.nc'
    PRECIP_DATA_DIR = os.path.join(BASE_DATA_DIR, PRECIP_SUBDIR)

    BASELINE_YEARS = range(1991, 2006) # Climatology baseline
    FWI_EXTREME_PERCENTILE = 95.0 # Percentile for defining extreme FWI days


    # REGION_NAME = 'US Southwest' 
    # REGION_COORDS = {
    #     'north': 35.5,  # Northern boundary
    #     'south': 29.0,  # Southern boundary
    #     'west': -122.0,  # Western boundary
    #     'east': -103.0   # Eastern boundary
    # }

    REGION_NAME = 'Southwest US & Northern Mexico'
    REGION_COORDS = {
            'north': 32.5,
            'south': 24.0,
            'west': -111.0,
            'east': -100.0
        }

    # REGION_NAME = "Southern California"
    # REGION_COORDS = {
    #     'north': 35.5,
    #     'south': 32.5,
    #     'west': -120.0,
    #     'east': -114.0
    # }

    # REGION_NAME = "California"
    # REGION_COORDS = {
    #     'north': 42.0,  # Northern boundary
    #     'south': 32.5,  # Southern boundary
    #     'west': -124.5,  # Western boundary
    #     'east': -114.0   # Eastern boundary
    # }

    # --- Load Data ---
    print("--- Loading Data ---")
    # Precipitation (Monthly)
    precip_zip_files = find_nc_files(PRECIP_DATA_DIR, PRECIP_FILE_PATTERN)
    precip_data = load_monthly_precip_from_zips(precip_zip_files)

    # print first few precipitation data values to see how big they are
    print("Before conversion, first few precipitation data values:")
    print(precip_data[PRECIP_VAR_NAME].isel(valid_time=0, latitude=slice(0,3), longitude=slice(0,3)).values)

    precip_data[PRECIP_VAR_NAME] = precip_data[PRECIP_VAR_NAME] * 1000.0
    precip_data[PRECIP_VAR_NAME].attrs['units'] = 'mm'

    # print first few precipitation data values to see how big they are after conversion
    print("After conversion, first few precipitation data values:")
    print(precip_data[PRECIP_VAR_NAME].isel(valid_time=0, latitude=slice(0,3), longitude=slice(0,3)).values)


    # --- Coordinate Handling & Region Mask ---
    print("\n--- Handling Coordinates & Region Mask ---")
    try:
        precip_lat_name, precip_lon_name = get_coord_names(precip_data)
        
        # For precipitation-only analysis, use the precip coordinates directly
        lat_name, lon_name = precip_lat_name, precip_lon_name
        
        # Check if coordinates need transformation based on precipitation data only
        # (No need to compare with FWI data)
        transform_region_coords = check_coordinate_system(precip_data, REGION_COORDS['west'], REGION_COORDS['east'])
        
        # Create region mask directly from precipitation data
        region_mask = create_bbox_mask(
            precip_data,
            REGION_COORDS['west'], REGION_COORDS['east'],
            REGION_COORDS['south'], REGION_COORDS['north'], 
            transform_region_coords
        )
        
        # Check if mask is valid
        if not region_mask.any().compute().item():
            raise ValueError(f"Precipitation mask for {REGION_NAME} is empty. Check coordinates.")
        print(f"Region mask created for {REGION_NAME}.")

    except Exception as e:
        print(f"Error during coordinate/mask handling: {e}")
        traceback.print_exc()
        return

    # --- Calculate Annual Metrics ---
    print(f"\n--- Calculating Annual Metrics for {REGION_NAME} ---")
    # 1. Precipitation Anomaly
    annual_precip_sums = calculate_annual_regional_sum(
        precip_data, PRECIP_VAR_NAME, region_mask, precip_lat_name, precip_lon_name
    )

    ytd_precip_sums, ytd_baseline_mean = calculate_ytd_regional_precipitation(
        precip_data, PRECIP_VAR_NAME, region_mask, BASELINE_YEARS, end_month=5, end_day=1, lat_name=precip_lat_name, lon_name=precip_lon_name
    )

    baseline_selection = annual_precip_sums.loc[annual_precip_sums.index.isin(BASELINE_YEARS)]
    baseline_precip_mean = baseline_selection.mean()
    print(f"  Baseline Annual Precipitation Mean (1991-2005): {baseline_precip_mean:.2f} mm")
    precip_annual_anomalies = annual_precip_sums - baseline_precip_mean
    precip_annual_pct_anomalies = (precip_annual_anomalies / baseline_precip_mean) * 100.0
    precip_annual_std = np.std([v for y, v in annual_precip_sums.items() if y in BASELINE_YEARS])
    precip_zscore_anomalies = (annual_precip_sums - baseline_precip_mean) / precip_annual_std

    # Make a copy of the annual anomalies
    combined_anomalies = precip_annual_anomalies.copy()
    combined_pct_anomalies = precip_annual_pct_anomalies.copy()
    combined_std_anomalies = precip_zscore_anomalies.copy()

    # If we have 2025 data, replace or add the YTD value for 2025
    if 2025 in ytd_precip_sums.index:
        ytd_anomaly_2025 = ytd_precip_sums[2025] - ytd_baseline_mean
        ytd_anomaly_pct_2025 = (ytd_anomaly_2025 / ytd_baseline_mean) * 100.0
        ytd_baseline_std = np.std([v for y, v in ytd_precip_sums.items() if y in BASELINE_YEARS])
        ytd_zscore_2025 = (ytd_precip_sums[2025] - ytd_baseline_mean) / ytd_baseline_std
        combined_anomalies[2025] = ytd_anomaly_2025
        combined_pct_anomalies[2025] = ytd_anomaly_pct_2025
        combined_std_anomalies[2025] = ytd_zscore_2025
        print(f"  2025 YTD Precipitation Anomaly (Jan 1-June 1): {ytd_anomaly_2025:.2f} mm")

    plot_annual_precip_anomalies(
        combined_std_anomalies,  # Use z-scores instead of percent anomalies
        REGION_NAME,
        OUTPUT_DIR,
        current_year=2025,
        ytd_end_month=6,
        ytd_end_day=1,
        use_zscore=True  # Indicate we're using z-scores
    )

     # =====================
    # FRENCH LANGUAGE CHARTS
    # =====================
    print("\nGenerating French language precipitation charts...")
    
    # Generate French version of precipitation anomalies
    plot_annual_precip_anomalies_french(
        combined_std_anomalies,
        REGION_NAME,
        OUTPUT_DIR,
        current_year=2025,
        ytd_end_month=6,
        ytd_end_day=1,
        use_zscore=True
    )

    print("French language precipitation charts generated successfully!")

if __name__ == "__main__":
    main()