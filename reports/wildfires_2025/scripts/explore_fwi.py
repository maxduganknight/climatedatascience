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
import datetime
import geopandas as gpd
import shapely.geometry
from shapely.geometry import Polygon
import requests
import tempfile
import matplotlib.patheffects as path_effects
import gc

from utils import (
    setup_space_mono_font, setup_enhanced_plot, format_plot_title, 
    add_deep_sky_branding, save_plot, COLORS
)

# Suppress common warnings during calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings('ignore', category=UserWarning, message="No index created for dimension dayofyear")

SHAPEFILE_DIR = '../data/shapefiles/downloaded'

#######################
# DATA LOADING FUNCTIONS
#######################

def find_nc_files(data_dir, file_pattern):
    """Find NetCDF files matching the pattern in the given directory"""
    full_pattern = os.path.join(data_dir, file_pattern)
    matching_files = glob.glob(full_pattern)
    return matching_files

def load_nc_fwi_files(matching_files):
    """Load FWI data from the matching files"""
    print("\nLoading data from matching files...")
    # Explicitly specify the engine
    data = xr.open_mfdataset(matching_files, combine='by_coords', engine='netcdf4')
    
    # Print basic dataset info
    print("\nDataset dimensions:")
    print(f"Dimensions: {data.dims}")
    print("\nTime range:")
    time_min = pd.to_datetime(data.valid_time.min().values).strftime('%Y-%m-%d')
    time_max = pd.to_datetime(data.valid_time.max().values).strftime('%Y-%m-%d')
    print(f"From {time_min} to {time_max}")
    
    return data

def get_country_gdf(country_code='CAN', output_dir='../../data/shapefiles/downloaded'):
    """Download country shapefile and return as GeoDataFrame"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to save/load the file
    output_path = os.path.join(output_dir, f"{country_code.lower()}_borders.geojson")
    
    # Check if we already have the file
    if os.path.exists(output_path):
        print(f"Loading {country_code} shapefile from local cache: {output_path}")
        return gpd.read_file(output_path)
    
    print(f"Downloading {country_code} shapefile...")
    
    # Keep only the source that worked reliably
    source_url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson"
    
    try:
        # Download world dataset once and filter for countries
        tmp_file = os.path.join(output_dir, "world_countries.geojson")
        
        if not os.path.exists(tmp_file):
            print(f"Downloading world dataset...")
            response = requests.get(source_url)
            if response.status_code == 200:
                with open(tmp_file, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download: HTTP {response.status_code}")
        
        # Read the file and filter for our country
        world = gpd.read_file(tmp_file)
        
        # Define country name mapping
        country_names = {
            'CAN': 'Canada',
            'USA': 'United States of America', 
            'MEX': 'Mexico'
        }
        
        # Find country by name
        if country_code in country_names and 'name' in world.columns:
            country = world[world['name'] == country_names[country_code]]
            if not country.empty:
                print(f"Found {country_code} in world dataset!")
                country.to_file(output_path, driver='GeoJSON')
                return country
        
        raise Exception(f"Country {country_code} not found in dataset")
            
    except Exception as e:
        print(f"Error downloading shapefile: {e}")
        # Create fallback geometry
        fallbacks = {
            'CAN': [(-141, 41.7), (-141, 83), (-52.6, 83), (-52.6, 41.7), (-141, 41.7)],
            'USA': [(-125, 24), (-125, 49.5), (-66, 49.5), (-66, 24), (-125, 24)],
            'MEX': [(-118.5, 14.5), (-118.5, 32.5), (-86.5, 32.5), (-86.5, 14.5), (-118.5, 14.5)]
        }
        
        coords = fallbacks.get(country_code, [(-180, -90), (-180, 90), (180, 90), (180, -90)])
        poly = Polygon(coords)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        gdf['name'] = country_names.get(country_code, country_code)
        gdf.to_file(output_path, driver='GeoJSON')
        return gdf

#######################
# COORDINATE HANDLING FUNCTIONS
#######################

def get_coord_names(dataset):
    """Extract coordinate names from dataset consistently"""
    lat_name = 'latitude' if 'latitude' in dataset.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in dataset.dims else 'lon'
    return lat_name, lon_name

def check_coordinate_system(fwi_data, bbox_minx, bbox_maxx):
    """Check for coordinate system mismatch and return transform flag"""
    lat_name, lon_name = get_coord_names(fwi_data)
    lon_min_fwi = fwi_data[lon_name].min().values.item()
    lon_max_fwi = fwi_data[lon_name].max().values.item()
    
    # If FWI data is in 0-360 longitude and shapefile is in -180 to 180
    transform_coords = lon_min_fwi >= 0 and lon_max_fwi > 180 and bbox_minx < 0
    return transform_coords

def transform_longitude(lon_value, transform=True):
    """Transform longitude between -180:180 and 0:360 systems"""
    if transform and lon_value < 0:
        return lon_value + 360
    return lon_value

def create_bbox_mask(data, minx, maxx, miny, maxy, transform_coords=False, padding=0):
    """Create bounding box mask for data"""
    lat_name, lon_name = get_coord_names(data)
    
    # Adjust coordinates if necessary
    if transform_coords:
        adj_minx = transform_longitude(minx)
        adj_maxx = transform_longitude(maxx)
    else:
        adj_minx, adj_maxx = minx, maxx
    
    # Create mask
    bbox_mask = (
        (data[lon_name] >= adj_minx - padding) &
        (data[lon_name] <= adj_maxx + padding) &
        (data[lat_name] >= miny - padding) &
        (data[lat_name] <= maxy + padding)
    )
    
    return bbox_mask

def create_country_mask(fwi_data, country_gdf, transform_coords=False):
    """Create a mask for a specific country using its shapefile geometry"""
    print("Creating precise country mask from shapefile...")
    
    # Get coordinate names
    lat_name, lon_name = get_coord_names(fwi_data)
    
    # Dissolve country features into a single polygon with careful buffer
    # Use a small buffer to ensure proper topology
    country_polygon = country_gdf.dissolve().geometry.iloc[0].buffer(0.01)
    print(f"Created country polygon with area: {country_polygon.area:.2f} square degrees")
    
    # Create meshgrid of lat/lon points
    lons = fwi_data[lon_name].values
    lats = fwi_data[lat_name].values
    
    # Handle coordinate system transformation if needed
    if transform_coords:
        print("Adjusting coordinates for shapefile compatibility...")
        # Transform coordinates if FWI data is in 0-360 but shapefile is in -180-180
        adjusted_lons = np.where(lons > 180, lons - 360, lons)
    else:
        adjusted_lons = lons
    
    print(f"Coordinate ranges for testing: Lon {adjusted_lons.min():.2f} to {adjusted_lons.max():.2f}, Lat {lats.min():.2f} to {lats.max():.2f}")
    
    # Create a faster approach using a prepared geometry
    from shapely.prepared import prep
    prepared_polygon = prep(country_polygon)
    
    # Create matrix of same shape as the data initialized with False
    mask_shaped = np.zeros((len(lats), len(adjusted_lons)), dtype=bool)
    
    # Test points in smaller batches to avoid memory issues
    print("Testing points against country boundary (batch processing)...")
    step = 10  # Process points in a grid with this step size
    
    for i in range(0, len(lats), step):
        i_end = min(i + step, len(lats))
        for j in range(0, len(adjusted_lons), step):
            j_end = min(j + step, len(adjusted_lons))
            
            # Create a grid of points for this batch
            lon_subset, lat_subset = np.meshgrid(
                adjusted_lons[j:j_end], 
                lats[i:i_end]
            )
            
            # Test each point in this batch
            for ii in range(lon_subset.shape[0]):
                for jj in range(lon_subset.shape[1]):
                    point = shapely.geometry.Point(lon_subset[ii, jj], lat_subset[ii, jj])
                    mask_shaped[i+ii, j+jj] = prepared_polygon.contains(point)
    
    print(f"Point testing complete: {np.sum(mask_shaped)} points inside country boundary")
    
    # If we still have no points, try with a larger buffer as fallback
    if np.sum(mask_shaped) == 0:
        print("WARNING: No points found inside country. Trying with larger buffer...")
        # Try with a larger buffer
        country_polygon = country_gdf.dissolve().geometry.iloc[0].buffer(0.1)
        prepared_polygon = prep(country_polygon)
        
        # Simplify by just checking if point is in bounding box first
        minx, miny, maxx, maxy = country_polygon.bounds
        
        for i in range(len(lats)):
            for j in range(len(adjusted_lons)):
                lon, lat = adjusted_lons[j], lats[i]
                
                # First fast check if point is in bounding box
                if minx <= lon <= maxx and miny <= lat <= maxy:
                    # Then do exact check
                    point = shapely.geometry.Point(lon, lat)
                    mask_shaped[i, j] = prepared_polygon.contains(point)
        
        print(f"Fallback approach: {np.sum(mask_shaped)} points inside country boundary")
    
    # Convert to xarray DataArray with the same coordinates as the input data
    mask = xr.DataArray(
        mask_shaped, 
        coords={lat_name: lats, lon_name: lons}, 
        dims=[lat_name, lon_name]
    )
    
    print(f"Created country mask with {mask.sum().item()} grid cells inside the country")
    
    return mask

#######################
# DATA ANALYSIS FUNCTIONS
#######################

def calculate_fwi_anomaly(fwi_data, days_back=14, baseline_years=range(1991, 2021), return_all=True):
    """Calculate FWI anomalies for the most recent period compared to baseline years
    
    Parameters:
    -----------
    fwi_data : xarray.Dataset
        Dataset containing FWI data with 'valid_time' dimension and 'fwinx' variable
    days_back : int
        Number of days back from the most recent date to calculate the anomaly
    baseline_years : range or list
        Range of years to use as baseline for comparison
    return_all : bool
        Whether to return all three anomaly types (raw, percentage, and standard deviation)
        
    Returns:
    --------
    tuple
        (fwi_anomaly_raw, fwi_anomaly_pct, fwi_anomaly_std, start_date, end_date, month_day_start, month_day_end)
        where fwi_anomaly_std is the anomaly in units of standard deviation
    """
    # Calculate the most recent period
    end_date = pd.to_datetime(fwi_data.valid_time.max().values)
    start_date = end_date - pd.Timedelta(days=days_back)
    
    print(f"\nCalculating FWI anomalies:")
    print(f"Current period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Baseline period: 1991-2020 (same calendar window)")
    
    # Select data for the current period
    current_period = fwi_data.sel(valid_time=slice(start_date, end_date))
    current_fwi = current_period['fwinx'].mean(dim='valid_time')
    
    # Create mask for the same calendar days across the baseline years
    month_day_start = (start_date.month, start_date.day)
    month_day_end = (end_date.month, end_date.day)
    
    # Create a mask for baseline years
    baseline_mask = (
        ((fwi_data.valid_time.dt.month == month_day_start[0]) & 
            (fwi_data.valid_time.dt.day >= month_day_start[1])) | 
        ((fwi_data.valid_time.dt.month > month_day_start[0]) & 
            (fwi_data.valid_time.dt.month < month_day_end[0])) |
        ((fwi_data.valid_time.dt.month == month_day_end[0]) & 
            (fwi_data.valid_time.dt.day <= month_day_end[1]))
    ) & (fwi_data.valid_time.dt.year.isin(baseline_years))
    
    # Select data for the baseline period
    baseline_period = fwi_data.sel(valid_time=baseline_mask)
    
    # Group by day of year and calculate climatology
    baseline_period['dayofyear'] = baseline_period.valid_time.dt.dayofyear
    climatology = baseline_period.groupby('dayofyear').mean(dim='valid_time')
    
    # Also calculate standard deviation for standardized anomaly
    std_dev = baseline_period.groupby('dayofyear').std(dim='valid_time')
    
    # Calculate mean FWI for the same calendar window as the current period
    current_dayofyear_start = start_date.dayofyear
    current_dayofyear_end = end_date.dayofyear
    
    if current_dayofyear_start <= current_dayofyear_end:
        # Regular case, all days within same year
        baseline_days = range(current_dayofyear_start, current_dayofyear_end + 1)
    else:
        # Case when period spans across year end
        baseline_days = list(range(current_dayofyear_start, 367)) + list(range(1, current_dayofyear_end + 1))
    
    # Select days from climatology that match our window
    baseline_fwi = climatology.sel(dayofyear=baseline_days)['fwinx'].mean(dim='dayofyear')
    baseline_std = std_dev.sel(dayofyear=baseline_days)['fwinx'].mean(dim='dayofyear')
    
    # 1. Calculate raw (absolute) anomaly (current - baseline)
    fwi_anomaly_raw = current_fwi - baseline_fwi
    
    # Print some statistics about the raw anomaly
    print("\nFWI raw anomaly statistics:")
    print(f"Min: {fwi_anomaly_raw.min().values.item():.4f}")
    print(f"Max: {fwi_anomaly_raw.max().values.item():.4f}")
    print(f"Mean: {fwi_anomaly_raw.mean().values.item():.4f}")
    print(f"Std Dev: {fwi_anomaly_raw.std().values.item():.4f}")
    
    # Initialize other anomaly types
    fwi_anomaly_pct = None
    fwi_anomaly_std = None
    
    if return_all:
        # 2. Calculate percentage (relative) anomaly: ((current - baseline) / baseline) * 100
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Handle division by zero and very small baseline values
            # First identify problematic areas - where baseline is very small or zero
            small_baseline = baseline_fwi < 0.5  # Areas with FWI baseline near zero
            
            # For areas with reasonable baseline values, calculate proper percentage
            fwi_anomaly_pct = ((current_fwi - baseline_fwi) / baseline_fwi) * 100
            
            # Cap extreme values for better visualization
            max_pct = 500  # Cap at 500% increase
            min_pct = -100  # Cap at 100% decrease (can't go below -100%)
            
            # Apply caps to extreme values
            fwi_anomaly_pct = xr.where(fwi_anomaly_pct > max_pct, max_pct, fwi_anomaly_pct)
            fwi_anomaly_pct = xr.where(fwi_anomaly_pct < min_pct, min_pct, fwi_anomaly_pct)
            
            # Special handling for areas with very low baseline FWI
            # If baseline is very small but current is also low, set to 0% change
            # If baseline is very small but current is high, set to max_pct
            fwi_anomaly_pct = xr.where(
                small_baseline & (current_fwi < 3), 
                0,  # Areas with negligible FWI in both periods
                fwi_anomaly_pct
            )
            fwi_anomaly_pct = xr.where(
                small_baseline & (current_fwi >= 3),
                max_pct,  # Areas with high current FWI but historically very low
                fwi_anomaly_pct
            )
        
        # Print some statistics about the percentage anomaly
        print("\nFWI percentage anomaly statistics:")
        print(f"Min: {fwi_anomaly_pct.min().values.item():.2f}%")
        print(f"Max: {fwi_anomaly_pct.max().values.item():.2f}%")
        print(f"Mean: {fwi_anomaly_pct.mean().values.item():.2f}%")
        
        # 3. Calculate standard anomaly: (current - baseline) / standard_deviation
        # Improved standard anomaly calculation with better NaN handling
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Force computation to regular arrays to eliminate Dask warning issues
            if hasattr(current_fwi, 'compute'):
                current_np = current_fwi.compute().values
                baseline_np = baseline_fwi.compute().values
                std_np = baseline_std.compute().values
            else:
                current_np = current_fwi.values
                baseline_np = baseline_fwi.values
                std_np = baseline_std.values
                
            # Create mask for valid data points
            valid_mask = (~np.isnan(current_np)) & (~np.isnan(baseline_np)) & (~np.isnan(std_np))
            
            # Create mask for reasonable std values
            min_std_threshold = 1.0
            valid_std_mask = (std_np >= min_std_threshold) & valid_mask
            
            # Initialize result array
            std_anomaly_np = np.full_like(current_np, np.nan)
            
            # Where std is valid, calculate standard anomaly
            raw_anomaly_np = current_np - baseline_np
            std_anomaly_np[valid_std_mask] = raw_anomaly_np[valid_std_mask] / std_np[valid_std_mask]
            
            # Where std is too small but data is valid, use scaled raw anomaly
            low_std_mask = valid_mask & (~valid_std_mask)
            
            # Clip raw anomaly to reasonable range, then scale it to approximate std units
            clipped_anomaly = np.clip(raw_anomaly_np[low_std_mask], -10, 10)
            std_anomaly_np[low_std_mask] = clipped_anomaly / 2.5
            
            # Cap extreme values
            std_anomaly_np = np.clip(std_anomaly_np, -4.0, 4.0)
            
            # Convert back to xarray with original coordinates
            fwi_anomaly_std = xr.DataArray(
                std_anomaly_np,
                coords=current_fwi.coords,
                dims=current_fwi.dims
            )
        
        # Print statistics about the standard anomaly
        print("\nFWI standard anomaly statistics:")
        print(f"Min: {np.nanmin(std_anomaly_np):.2f} σ")
        print(f"Max: {np.nanmax(std_anomaly_np):.2f} σ")
        print(f"Mean: {np.nanmean(std_anomaly_np):.2f} σ")
    
    return fwi_anomaly_raw, fwi_anomaly_pct, fwi_anomaly_std, start_date, end_date, month_day_start, month_day_end

def calculate_fwi_percentiles(fwi_data, baseline_years=range(1991, 2021)):
    """Calculate 95th percentile FWI values for the baseline period"""
    print("\nCalculating 95th percentile FWI values for baseline period (1991-2020)...")
    
    # Select data for the baseline period
    baseline_mask = fwi_data.valid_time.dt.year.isin(baseline_years)
    baseline_period = fwi_data.sel(valid_time=baseline_mask)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # Calculate 95th percentile along the time dimension
        fwi_95th_percentile = baseline_period['fwinx'].quantile(0.95, dim='valid_time')
    
    # Get valid min and max safely
    valid_data = fwi_95th_percentile.values[~np.isnan(fwi_95th_percentile.values)]
    if len(valid_data) > 0:
        min_val = float(valid_data.min())
        max_val = float(valid_data.max())
        print(f"95th percentile calculation complete")
        print(f"Min 95th percentile value: {min_val:.2f}")
        print(f"Max 95th percentile value: {max_val:.2f}")
    else:
        print("WARNING: No valid data points found in 95th percentile calculation")
        min_val = 0
        max_val = 100
    
    # Calculate non-NaN percentage for completeness assessment
    total_cells = fwi_95th_percentile.size
    valid_cells = (~np.isnan(fwi_95th_percentile.values)).sum()
    valid_percentage = (valid_cells / total_cells) * 100 if total_cells > 0 else 0
    
    print(f"Data coverage: {valid_percentage:.1f}% of grid cells have valid percentile values")
    
    return fwi_95th_percentile

def get_current_period_data(fwi_data, days_back=30):
    """Get data for the most recent period"""
    end_date = pd.to_datetime(fwi_data.valid_time.max().values)
    start_date = end_date - pd.Timedelta(days=days_back)
    
    current_mask = (fwi_data.valid_time >= np.datetime64(start_date)) & (fwi_data.valid_time <= np.datetime64(end_date))
    current_data = fwi_data.sel(valid_time=current_mask)
    current_fwi = current_data['fwinx'].mean(dim='valid_time')
    
    return current_fwi, start_date, end_date

def calculate_state_fwi_anomaly(fwi_anomaly, state_name, state_gdf):
    """Calculate FWI anomaly statistics for a specific state using shapefile masking"""
    print(f"\nCalculating FWI anomaly for {state_name}...")
    
    # Filter to just the requested state
    state = state_gdf[state_gdf['NAME'] == state_name]
    if state.empty:
        print(f"Error: State '{state_name}' not found in shapefile.")
        return None, None
    
    # Get coordinate info
    lat_name, lon_name = get_coord_names(fwi_anomaly)
    minx, miny, maxx, maxy = state.total_bounds
    
    # Print debugging information
    print(f"FWI data coordinates:")
    print(f"  {lon_name} range: {fwi_anomaly[lon_name].min().values.item():.4f} to {fwi_anomaly[lon_name].max().values.item():.4f}")
    print(f"  {lat_name} range: {fwi_anomaly[lat_name].min().values.item():.4f} to {fwi_anomaly[lat_name].max().values.item():.4f}")
    print(f"State boundary box:")
    print(f"  Longitude: {minx:.4f} to {maxx:.4f}")
    print(f"  Latitude: {miny:.4f} to {maxy:.4f}")
    print(f"Shapefile CRS: {state_gdf.crs}")
    
    # Check for coordinate system mismatch
    transform_coords = check_coordinate_system(fwi_anomaly, minx, maxx)
    if transform_coords:
        print("Coordinate system mismatch detected: FWI uses 0-360 longitude, shapefile uses -180 to 180")
        # Convert -180 to 180 coordinates to 0 to 360
        adj_minx = transform_longitude(minx, transform_coords)
        adj_maxx = transform_longitude(maxx, transform_coords)
        print(f"Adjusted boundary box:")
        print(f"  Longitude: {adj_minx:.4f} to {adj_maxx:.4f}")
    
    try:
        # Create mask for the state
        bbox_mask = create_bbox_mask(
            fwi_anomaly, minx, maxx, miny, maxy, transform_coords
        )
        
        # Apply the mask to the anomaly data
        masked_anomaly = fwi_anomaly.where(bbox_mask)
        
        # Check if we have any valid data
        valid_data_count = (~np.isnan(masked_anomaly.values)).sum()
        print(f"Number of valid data points after masking: {valid_data_count}")
        
        if valid_data_count == 0:
            print("WARNING: No valid data points found after masking!")
            # Try with a slightly expanded bounding box as a fallback
            print("Attempting with a larger boundary box (adding 1 degree padding)...")
            expanded_bbox_mask = create_bbox_mask(
                fwi_anomaly, minx, maxx, miny, maxy, transform_coords, padding=1
            )
            
            masked_anomaly = fwi_anomaly.where(expanded_bbox_mask)
            valid_data_count = (~np.isnan(masked_anomaly.values)).sum()
            print(f"Number of valid data points after expanded masking: {valid_data_count}")
        
        # Compute to convert from Dask to NumPy arrays
        masked_anomaly = masked_anomaly.compute()
        
    except Exception as e:
        print(f"Error creating state mask: {e}")
        return None, None
    
    # Calculate statistics
    state_stats = calculate_masked_statistics(masked_anomaly, state_name, valid_data_count, state)
    
    # Print statistics
    print(f"\nFWI anomaly statistics for {state_name}:")
    print(f"Mean: {state_stats['mean']:.4f}")
    print(f"Median: {state_stats['median']:.4f}")
    print(f"Min: {state_stats['min']:.4f}")
    print(f"Max: {state_stats['max']:.4f}")
    print(f"Std Dev: {state_stats['std']:.4f}")
    
    return state_stats, masked_anomaly

def calculate_masked_statistics(masked_data, name, valid_data_count=None, state_gdf_row=None):
    """Helper function to calculate statistics from masked data"""
    if valid_data_count is None:
        valid_data_count = (~np.isnan(masked_data.values)).sum()
    
    # Handle empty data case
    if valid_data_count == 0:
        print("WARNING: No valid data to calculate statistics!")
        stats = {
            'name': name,
            'min': float('nan'),
            'max': float('nan'),
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan')
        }
        return stats
    
    # Calculate statistics with proper error handling
    try:
        lat_name, lon_name = get_coord_names(masked_data)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            stats = {
                'name': name,
                'min': float(masked_data.min().values),
                'max': float(masked_data.max().values),
                'mean': float(masked_data.mean().values),
                'std': float(masked_data.std().values)
            }
        
        # Handle median calculation carefully
        try:
            stats['median'] = float(masked_data.median(dim=[lat_name, lon_name]).values)
        except:
            try:
                stats['median'] = float(np.nanmedian(masked_data.values))
            except:
                stats['median'] = float('nan')
        
        # Add area if available in the shapefile
        if state_gdf_row is not None and 'ALAND' in state_gdf_row.columns:
            stats['area_sqkm'] = float(state_gdf_row['ALAND'].iloc[0] / 1e6)  # Convert from sq meters to sq km
        
        return stats
    
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        stats = {
            'name': name,
            'min': float('nan'),
            'max': float('nan'),
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan')
        }
        return stats

def calculate_extreme_areas(current_fwi, fwi_95th_local):
    """Calculate where current FWI exceeds the 95th percentile"""
    # Find where current FWI exceeds the 95th percentile
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        extreme_mask = current_fwi > fwi_95th_local

    # Force compute if dask arrays - Moved earlier to ensure numpy inputs below
    if hasattr(extreme_mask, 'compute'):
        extreme_mask = extreme_mask.compute()
    if hasattr(current_fwi, 'compute'):
        current_fwi = current_fwi.compute()

    # Calculate percentage area exceeding threshold using numpy values
    current_fwi_values = current_fwi.values
    extreme_mask_values = extreme_mask.values

    total_cells = (~np.isnan(current_fwi_values)).sum()
    if total_cells == 0:
        # Return float 0.0 for percentage
        return extreme_mask, 0, 0.0

    extreme_cells = np.sum(extreme_mask_values & ~np.isnan(current_fwi_values))
    # Ensure float division and explicitly cast the result to float
    extreme_pct = float(extreme_cells) / float(total_cells) * 100.0

    # Return float for percentage
    return extreme_mask, int(extreme_cells), float(extreme_pct)

#######################
# VISUALIZATION FUNCTIONS
#######################

def create_custom_colormap(name='fire_weather_diverging', anomaly_type='raw'):
    """Create a single, consistent colormap with type-specific enhancements and more dramatic coloring"""
    if anomaly_type == 'std':
        # Enhanced colormap specifically for standard deviations with proper blue-white-red progression
        colors = [
            (0.0, 0.0, 0.4),     # Dark navy blue (most negative)
            (0.0, 0.2, 0.6),     # Dark blue
            (0.2, 0.4, 0.8),     # Medium blue
            (0.4, 0.6, 0.9),     # Light blue
            (0.7, 0.8, 1.0),     # Very light blue
            (0.9, 0.9, 0.95),    # Near white (blue tint)
            (1.0, 1.0, 1.0),     # Pure white at center (zero)
            (0.95, 0.9, 0.9),    # Near white (red tint)
            (1.0, 0.8, 0.7),     # Very light red
            (1.0, 0.6, 0.4),     # Light red
            (0.9, 0.4, 0.2),     # Medium red
            (0.8, 0.2, 0.0),     # Dark red
            (0.6, 0.0, 0.0),     # Deep red
            (0.4, 0.0, 0.0)      # Dark crimson (most positive)
        ]
        
        # For standard deviation, use evenly spaced nodes with white exactly at center
        positions = [
            0.0,    # Most negative (-3σ)
            0.1,    # -2.5σ
            0.2,    # -2σ
            0.3,    # -1.5σ
            0.4,    # -1σ
            0.45,   # -0.5σ
            0.5,    # 0σ (WHITE - exact center)
            0.55,   # +0.5σ
            0.6,    # +1σ
            0.7,    # +1.5σ
            0.8,    # +2σ
            0.85,   # +2.5σ
            0.9,    # +3σ
            1.0     # Most positive (>3σ)
        ]
        
        return plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            name, list(zip(positions, colors)), N=256)
    else:
        # Default colormap with dramatic gradient for raw and percentage anomalies
        colors = [
            (0, 0, 0.7),        # Deep blue
            (0.3, 0.3, 0.9),    # Medium blue
            (0.6, 0.6, 1.0),    # Light blue
            (0.8, 0.8, 1.0),    # Very light blue
            (0.95, 0.95, 0.95), # Near white (blue side)
            (1.0, 1.0, 1.0),    # Pure white for neutral (EXACTLY at center)
            (0.95, 0.95, 0.95), # Near white (red side)
            (1.0, 0.8, 0.7),    # Very light pink
            (1.0, 0.6, 0.4),    # Light orange-red
            (1.0, 0.4, 0.2),    # Orange-red
            (1.0, 0.2, 0.0),    # Bright red
            (0.8, 0.1, 0.0),    # Deep red
            (0.6, 0.05, 0.0),   # Crimson
            (0.4, 0.05, 0.0),   # Dark burgundy 
            (0.2, 0.02, 0.0),   # Dark brown
            (0.0, 0.0, 0.0)     # Pure black
        ]

        # For raw anomaly, explicitly position the white color at the center (0.5)
        # and cluster more colors toward the positive end for better visualization of extremes
        positions = [
            0.0,    # Deep blue
            0.15,   # Medium blue
            0.25,   # Light blue
            0.35,   # Very light blue
            0.45,   # Near white (blue side)
            0.5,    # Pure white (EXACTLY at center)
            0.55,   # Near white (red side)
            0.6,    # Very light pink
            0.65,   # Light orange-red  
            0.7,    # Orange-red
            0.75,   # Bright red
            0.8,    # Deep red
            0.85,   # Crimson
            0.9,    # Dark burgundy
            0.95,   # Dark brown
            1.0     # Pure black
        ]
        
        # Create colormap with explicit positions for each color
        return plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            name, list(zip(positions, colors)), N=256)

def setup_map_figure(figsize=(15, 10), projection_type="albers"):
    """Set up a figure with cartopy for mapping with improved projection"""
    plt.figure(figsize=figsize)
    
    if (projection_type == "albers"):
        # Albers Equal Area projection - good for North America with nice curvature
        projection = ccrs.AlbersEqualArea(central_longitude=-96.0, central_latitude=37.5,
                                         standard_parallels=(29.5, 45.5))
    elif (projection_type == "lambert"):
        # Lambert Conformal projection - another good option with visible curvature
        projection = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5,
                                          standard_parallels=(33, 45))
    else:
        # Fall back to PlateCarree for simple cases
        projection = ccrs.PlateCarree()
    
    ax = plt.axes(projection=projection)
    
    # Add basic features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.4)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.7, edgecolor='gray', alpha=0.8)
    
    return ax

def set_map_extent(ax, minx, maxx, miny, maxy, padding=0.5, transform_coords=False):
    """Set the extent of the map"""
    # Handle coordinate system for display
    if transform_coords:
        cartopy_minx = minx  # For display, use original coordinates
        cartopy_maxx = maxx
    else:
        cartopy_minx = minx
        cartopy_maxx = maxx
        
    ax.set_extent([cartopy_minx-padding, cartopy_maxx+padding, 
                 miny-padding, maxy+padding], crs=ccrs.PlateCarree())

def add_colorbar(ax, mappable, label, horizontal=True, extend='both'):
    """Add a colorbar to the plot"""
    if horizontal:
        cbar = plt.colorbar(mappable, ax=ax, orientation='horizontal', 
                          shrink=0.8, pad=0.05, extend=extend)
    else:
        cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', 
                          shrink=0.8, pad=0.02, extend=extend)
    
    cbar.set_label(label, fontsize=12, weight='bold')
    
    # Add more tick marks for better readability
    tick_locator = plt.matplotlib.ticker.MaxNLocator(nbins=9)
    cbar.locator = tick_locator
    cbar.update_ticks()
    
    return cbar

def add_gridlines(ax, show_gridlines=False):
    """Add gridlines to the map if requested"""
    if show_gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
        gl.top_labels = False
        gl.right_labels = False
        return gl
    return None

# def plot_fwi_anomaly_map(fwi_anomaly, start_date, end_date, output_file='fwi_anomaly_map.png', colorbar_label='Fire Weather Index Anomaly'):
#     """Create a map visualization of FWI anomaly with enhanced styling"""
#     # Set up the basic map with cartopy
#     ax = setup_map_figure()
    
#     # Add only US states, not countries
#     ax.add_feature(cfeature.STATES, linewidth=0.7, edgecolor='gray', alpha=0.8)
    
#     # Set map extent to continental US only (tighter bounds)
#     ax.set_extent([-140, -50, 14, 60], crs=ccrs.PlateCarree())

#     # Create custom colormap 
#     custom_cmap = create_custom_colormap(name='custom_diverging')
    
#     # Calculate the max absolute value for a symmetric color scale
#     vmax = max(abs(fwi_anomaly.min().values.item()), abs(fwi_anomaly.max().values.item()))
#     contrast_factor = 1.0  # Increase for more contrast
#     norm = TwoSlopeNorm(vmin=-vmax*contrast_factor, vcenter=0, vmax=vmax*contrast_factor)
    
#     # Plot the anomaly data
#     im = fwi_anomaly.plot(ax=ax, cmap=custom_cmap, norm=norm, 
#                          transform=ccrs.PlateCarree(),
#                          add_colorbar=False)
    
#     # Add borders last for clarity
#     ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='gray')
    
#     # Add grid lines
#     add_gridlines(ax, show_gridlines=False)
    
#     # Add colorbar
#     cbar = add_colorbar(ax, im, colorbar_label)
    
#     # Annotate extreme values
#     if vmax > 10:
#         cbar.ax.text(0.95, 0.5, 'Extreme Fire Risk', 
#                     transform=cbar.ax.transAxes, 
#                     ha='right', va='center', 
#                     color='white', fontsize=10,
#                     bbox=dict(facecolor=COLORS['primary_dark'], alpha=0.7, boxstyle='round,pad=0.3'))
        
#         cbar.ax.text(0.05, 0.5, 'Very Low Risk', 
#                     transform=cbar.ax.transAxes, 
#                     ha='left', va='center', 
#                     color='white', fontsize=10,
#                     bbox=dict(facecolor='darkblue', alpha=0.7, boxstyle='round,pad=0.3'))
    
#     # Create figure and get font properties
#     fig = plt.gcf()
#     _, _, font_props = setup_enhanced_plot(figsize=fig.get_size_inches())
    
#     # Format plot title with consistent styling
#     title = 'U.S. FIRE WEATHER INDEX ANOMALY'
#     subtitle = f'Current: {start_date.strftime("%b %d")} - {end_date.strftime("%b %d, %Y")} vs. 1991-2020 Baseline'
#     format_plot_title(ax, title, subtitle, font_props)
    
#     # Add Deep Sky branding with data note
#     add_deep_sky_branding(ax, font_props, data_note="DATA: CEMS FIRE WEATHER INDEX")
    
#     # Get base filename without extension for multiple formats
#     base_output_file = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
    
#     # Save the plot with shared utility function
#     save_plot(fig, base_output_file + ".png")
#     print(f"\nEnhanced map saved as '{base_output_file}.png'")
    
#     # Save as SVG as well
#     save_plot(fig, base_output_file + ".svg")
#     print(f"\nVector version saved as '{base_output_file}.svg'")
    
#     plt.close()

def plot_state_extreme_fwi(fwi_data, state_name, state_gdf, days_back=30, output_file=None):
    """Create a map highlighting areas where current FWI exceeds the local 95th percentile"""
    if output_file is None:
        output_file = f'figures/fwi_extreme_{state_name.lower().replace(" ", "_")}.png'
    
    # Calculate the recent period data
    current_fwi, start_date, end_date = get_current_period_data(fwi_data, days_back)
    
    print(f"\nAnalyzing extreme Fire Weather Index for {state_name}")
    print(f"Current period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get the state boundary
    state = state_gdf[state_gdf['NAME'] == state_name]
    if state.empty:
        print(f"Error: State '{state_name}' not found in shapefile.")
        return
    
    # Get the bounding box with padding
    minx, miny, maxx, maxy = state.total_bounds
    padding = 0.5
    
    # Get coordinate names and check for coordinate system mismatch
    lat_name, lon_name = get_coord_names(fwi_data)
    transform_coords = check_coordinate_system(fwi_data, minx, maxx)
    
    if transform_coords:
        print("Adjusting coordinates to match FWI dataset (0-360° longitude)")
    
    # Create spatial mask and extract regional data
    bbox_mask = create_bbox_mask(fwi_data, minx, maxx, miny, maxy, transform_coords, padding)
    regional_data = fwi_data.where(bbox_mask, drop=True)
    
    # Calculate local 95th percentiles
    print("Calculating grid cell-specific 95th percentiles...")
    baseline_mask = (regional_data.valid_time.dt.year >= 1991) & (regional_data.valid_time.dt.year <= 2020)
    baseline_data = regional_data.sel(valid_time=baseline_mask)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        fwi_95th_local = baseline_data['fwinx'].quantile(0.95, dim='valid_time')
    
    # Calculate current period mean FWI for the region
    current_mask = (regional_data.valid_time >= np.datetime64(start_date)) & (regional_data.valid_time <= np.datetime64(end_date))
    current_data = regional_data.sel(valid_time=current_mask)
    current_regional_fwi = current_data['fwinx'].mean(dim='valid_time')
    
    # Identify extreme areas
    extreme_mask, extreme_cells, extreme_pct = calculate_extreme_areas(current_regional_fwi, fwi_95th_local)
    
    # Force computation from dask if needed
    if hasattr(current_regional_fwi, 'compute'):
        current_regional_fwi = current_regional_fwi.compute()
        fwi_95th_local = fwi_95th_local.compute()
    
    print(f"Analysis complete: {extreme_pct:.1f}% of {state_name} exceeds the local 95th percentile threshold")
    
    # Create the map
    ax = setup_map_figure()
    
    # Set map extent
    set_map_extent(ax, minx, maxx, miny, maxy, padding, transform_coords)
    
    # Plot base FWI values with gray colormap
    base_cmap = plt.cm.Greys_r
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        vmin = np.nanmin(current_regional_fwi.values)
        vmax = np.nanmax(current_regional_fwi.values)
    
    base_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    base_plot = current_regional_fwi.plot(ax=ax, cmap=base_cmap, norm=base_norm,
                                     transform=ccrs.PlateCarree(),
                                     add_colorbar=False, alpha=0.3)
    
    # Plot areas exceeding the threshold with vibrant colormap
    extreme_values = current_regional_fwi.where(extreme_mask)
    
    if extreme_cells > 0:
        # Custom colormap for extreme values
        extreme_colors = [(1, 0.7, 0),      # Orange
                         (1, 0.5, 0),       # Dark orange
                         (1, 0.3, 0),       # Red-orange
                         (1, 0.1, 0),       # Bright red
                         (0.8, 0, 0)]       # Dark red
        
        extreme_cmap = create_custom_colormap(extreme_colors, name='extreme_fire')
        
        extreme_plot = extreme_values.plot(ax=ax, cmap=extreme_cmap,
                                         transform=ccrs.PlateCarree(),
                                         add_colorbar=False)
        
        # Add colorbar for extreme values
        cbar_extreme = add_colorbar(ax, extreme_plot, 'Current FWI (Areas Exceeding Local 95th Percentile)')
    
    # Add state boundary on top
    state.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, transform=ccrs.PlateCarree())
    
    # Add grid lines
    add_gridlines(ax, show_gridlines=False)
    
    # Add title and caption
    plt.suptitle(f'Extreme Fire Weather Conditions: {state_name}', fontsize=16, weight='bold', y=0.98)
    plt.title(f'Areas exceeding the local 95th percentile FWI threshold (1991-2020 baseline)\n'
             f'Current period: {start_date.strftime("%b %d")} - {end_date.strftime("%b %d, %Y")}', 
             fontsize=14)
    
    # Add statistics text box
    textstr = (f"{extreme_pct:.1f}% of {state_name} currently exceeds the\n"
              f"local 95th percentile of historical fire weather conditions")
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)
    
    # Add source note
    plt.figtext(0.99, 0.01, "Data source: CEMS Fire Weather Index", 
               ha='right', fontsize=8, style='italic')
    
    # Save the map
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved as '{output_file}'")
    plt.close()

def plot_state_fwi_anomaly(masked_anomaly, state_name, state_gdf, start_date, end_date, output_file=None):
    """Create a map visualization of FWI anomaly for a specific state"""
    if output_file is None:
        output_file = f'fwi_anomaly_{state_name.lower().replace(" ", "_")}.png'
    
    # Get coordinate names
    lat_name, lon_name = get_coord_names(masked_anomaly)
    
    # Get the state boundary
    state = state_gdf[state_gdf['NAME'] == state_name]
    if state.empty:
        print(f"Error: State '{state_name}' not found in shapefile.")
        return
    
    # Get bounding box and set up map
    minx, miny, maxx, maxy = state.total_bounds
    padding = 0.5
    
    # Create the map
    ax = setup_map_figure()
    set_map_extent(ax, minx, maxx, miny, maxy, padding)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap(name='custom_diverging')
    
    # Calculate the max absolute value for a symmetric color scale
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        vmax = max(abs(masked_anomaly.min().values.item()), abs(masked_anomaly.max().values.item()))
    
    # Use a custom normalization
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot the anomaly data
    im = masked_anomaly.plot(ax=ax, cmap=custom_cmap, norm=norm, 
                           transform=ccrs.PlateCarree(),
                           add_colorbar=False)
    
    # Plot the state boundary
    state.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, transform=ccrs.PlateCarree())
    
    # Add grid lines
    add_gridlines(ax, show_gridlines=False)
    
    # Add colorbar
    add_colorbar(ax, im, 'Fire Weather Index Anomaly')
    
    # Add title and labels
    plt.suptitle(f'Fire Weather Index Anomaly: {state_name}', fontsize=16, weight='bold', y=0.98)
    plt.title(f'Current: {start_date.strftime("%b %d")} - {end_date.strftime("%b %d, %Y")} vs. 1991-2020 Baseline\n', 
             fontsize=14)
    
    # Add statistics in a text box
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        mean_val = masked_anomaly.mean().values.item()
        max_val = masked_anomaly.max().values.item()
    
    textstr = f"Mean anomaly: {mean_val:.2f}\nMax anomaly: {max_val:.2f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)
    
    # Add copyright/source note
    plt.figtext(0.99, 0.01, "Data source: CEMS Fire Weather Index", 
               ha='right', fontsize=8, style='italic')
    
    # Save the map
    plt.savefig(output_file, dpi=300)
    print(f"\nState map saved as '{output_file}'")
    plt.close()

def compare_state_fwi_anomalies(state_stats_list, start_date, end_date, output_file=None):
    """Create a bar chart comparing FWI anomalies across multiple states"""
    if output_file is None:
        output_file = f'state_fwi_anomaly_comparison.png'
    
    # Sort states by mean anomaly (descending)
    sorted_stats = sorted(state_stats_list, key=lambda x: x['mean'], reverse=True)
    
    # Extract data for plotting
    states = [s['name'] for s in sorted_stats]
    mean_anomalies = [s['mean'] for s in sorted_stats]
    
    # Determine colors based on anomaly values (red for positive, blue for negative)
    colors = ['firebrick' if val > 0 else 'steelblue' for val in mean_anomalies]
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(states, mean_anomalies, color=colors)
    
    # Add a horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Add value labels on top/bottom of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            y_pos = height + 0.1
            va = 'bottom'
        else:
            y_pos = height - 0.1
            va = 'top'
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}',
                ha='center', va=va)
    
    # Add labels and title
    plt.xlabel('State')
    plt.ylabel('Mean Fire Weather Index Anomaly')
    plt.title(f'FWI Anomaly Comparison by State\n{start_date.strftime("%b %d")} - {end_date.strftime("%b %d, %Y")} vs. 1991-2020 Baseline', 
             fontsize=14)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nState comparison chart saved as '{output_file}'")
    plt.close()

def get_fwi_for_year(fwi_data, year, coords, end_date, days_back=30):
    """Get mean FWI for a specific year and date range for a given area"""
    # Calculate the same window for the specified year
    year_end_date = datetime.datetime(year, end_date.month, end_date.day)
    year_start_date = year_end_date - pd.Timedelta(days=days_back)
    
    # Create a time mask for the specific year and date range
    time_mask = (
        (fwi_data.valid_time >= np.datetime64(year_start_date)) & 
        (fwi_data.valid_time <= np.datetime64(year_end_date))
    )
    
    # Check if we have any data for this time period
    year_data = fwi_data.sel(valid_time=time_mask)
    
    if year_data.sizes['valid_time'] == 0:
        print(f"No data found for {year_start_date} to {year_end_date}")
        return np.nan
    
    # Get coordinate names
    lat_name, lon_name = get_coord_names(year_data)
    
    # Handle possible longitude format differences (0-360 vs -180 to 180)
    lon_west = coords['west_bound']
    lon_east = coords['east_bound']
    
    # Convert to 0-360 format if needed
    if hasattr(year_data, lon_name) and year_data[lon_name].min() >= 0 and lon_west < 0:
        lon_west = transform_longitude(lon_west)
        lon_east = transform_longitude(lon_east)
    
    # Filter data for the specified area
    try:
        filtered_data = year_data.sel(
            {
                lat_name: slice(coords['north_bound'], coords['south_bound']),
                lon_name: slice(lon_west, lon_east)
            }
        )
        
        # Check if we have valid data after spatial filtering
        if filtered_data['fwinx'].isnull().all():
            print(f"All data is NaN in {year}")
            return np.nan
            
        # Calculate mean FWI for the area
        mean_fwi = filtered_data['fwinx'].mean(dim=[lat_name, lon_name]).mean(dim='valid_time').values.item()
        return mean_fwi
    except Exception as e:
        print(f"Error selecting data for {year}: {e}")
        return np.nan

def create_fwi_bar_chart(fwi_data, coords, end_date, days_back=90, output_file=None):
    """Create a bar chart showing FWI values across years for a specific area"""
    if output_file is None:
        output_file = f'{coords["name"].replace(" ", "_")}_fwi_by_year.png'
    
    # Get all years available in the dataset
    all_years = sorted(list(set(fwi_data.valid_time.dt.year.values)))

    # Calculate mean FWI for each year
    year_fwi_values = []
    for year in all_years:
        year_fwi = get_fwi_for_year(fwi_data, year, coords, end_date, days_back)
        year_fwi_values.append(year_fwi)

    # Filter out years with no data
    valid_years = [year for i, year in enumerate(all_years) if not np.isnan(year_fwi_values[i])]
    valid_fwi_values = [val for val in year_fwi_values if not np.isnan(val)]

    if valid_years:
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(valid_years, valid_fwi_values, color='steelblue')
        
        # Highlight the most recent year in a different color
        if valid_years and valid_years[-1] == all_years[-1]:
            bars[-1].set_color('firebrick')
        
        # Calculate month/day for display
        month_day_start = (end_date - pd.Timedelta(days=days_back)).strftime("%m/%d")
        month_day_end = end_date.strftime("%m/%d")
        
        # Add labels and title
        plt.xlabel('Year')
        plt.ylabel('Average Fire Weather Index (FWI)')
        plt.title(f'Average Fire Weather Index for {coords["name"]} Area\n'
                f'({month_day_start} - {month_day_end} window for each year)', 
                fontsize=14)
        
        # Add grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability if there are many years
        if len(valid_years) > 10:
            plt.xticks(rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"\nBar chart saved as '{output_file}'")
        plt.close()
    else:
        print(f"\nNo valid data found for {coords['name']} area across all years. Cannot create chart.")

def create_fwi_line_chart(fwi_data, coords, end_date, days_back=90, output_file=None, include_minimap=True):
    """Create a line chart showing FWI values across years for a specific area"""
    if output_file is None:
        output_file = f'{coords["name"].replace(" ", "_")}_fwi_line_by_year.png'
    
    # Get current year
    current_year = end_date.year
    
    # Define years to analyze (2015-2025 period)
    years_to_analyze = range(2015, current_year + 1)

    # Calculate mean FWI for each year
    year_fwi_values = []
    valid_years = []
    
    for year in years_to_analyze:
        year_fwi = get_fwi_for_year(fwi_data, year, coords, end_date, days_back)
        if not np.isnan(year_fwi):
            year_fwi_values.append(year_fwi)
            valid_years.append(year)
        else:
            print(f"No valid data for {coords['name']} in {year}")

    if valid_years:
        # Create line chart
        fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))
        
        # Plot line with markers
        plt.plot(valid_years, year_fwi_values, 'o-', 
                 color=COLORS['comparison'], 
                 linewidth=2.5, 
                 markersize=8)
        
        # Highlight the most recent year with a different color
        if valid_years and valid_years[-1] == current_year:
            plt.plot(valid_years[-1], year_fwi_values[-1], 'o', 
                    color=COLORS['primary'], 
                    markersize=10)
            
            # Add annotation for current year
            plt.annotate(f"{year_fwi_values[-1]:.1f}",
                        (valid_years[-1], year_fwi_values[-1]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        color=COLORS['primary'],
                        fontweight='bold',
                        fontproperties=font_props.get('bold'))
        
        # Calculate month/day for display
        month_day_start = (end_date - pd.Timedelta(days=days_back)).strftime("%b %d")
        month_day_end = end_date.strftime("%b %d")
        
        # Format plot title with consistent styling
        title = f"FIRE RISK IN {coords['name'].upper()} HAS REACHED RECORD LEVELS"
        subtitle = "AVERAGE SPRING FIRE WEATHER INDEX"
        format_plot_title(ax, title, subtitle, font_props)
        
        # Format x-axis to show all years
        plt.xticks(list(years_to_analyze))
        plt.xlim(min(years_to_analyze) - 0.5, max(years_to_analyze) + 0.5)
        
        
        # Apply font properties to ticks
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_props.get('regular'))
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_props.get('regular'))
            
        # Add Deep Sky branding with citation
        add_deep_sky_branding(ax, font_props, data_note=f"DATA: CEMS FIRE WEATHER INDEX. March 1 - June 1 average FWI.")
        
        # Add minimap if requested
        if include_minimap:
            fig = add_mini_map(fig, coords)
        
        # Save plot with consistent format
        save_plot(fig, output_file)
        print(f"\nLine chart saved as '{output_file}'")
        return fig
    else:
        print(f"\nNo valid data found for {coords['name']} area across years. Cannot create chart.")
        return None

def define_location_coords():
    """Define coordinates for various locations"""
    locations = {
        # 'albuquerque': {
        #     'name': 'Albuquerque, NM',
        #     'north_bound': 35.3,
        #     'south_bound': 34.8,
        #     'west_bound': -107.0,
        #     'east_bound': -106.3
        # },
        # 'santa_fe': {
        #     'name': 'Santa Fe, NM',
        #     'north_bound': 35.8,
        #     'south_bound': 35.5,
        #     'west_bound': -106.1,
        #     'east_bound': -105.8
        # },
        # 'el_paso': {
        #     'name': 'El Paso, TX',
        #     'north_bound': 32.0,
        #     'south_bound': 31.5,
        #     'west_bound': -107.0,
        #     'east_bound': -106.3
        # },
        # 'austin': {
        #     'name': 'Austin, TX',
        #     'north_bound': 30.4,
        #     'south_bound': 30.1,
        #     'west_bound': -98.0,
        #     'east_bound': -97.5
        # },
        # 'fort_worth': {
        #     'name': 'Fort Worth, TX',
        #     'north_bound': 32.8,
        #     'south_bound': 32.5,
        #     'west_bound': -98.0,
        #     'east_bound': -97.5
        # },
        # 'san_antonio': {
        #     'name': 'San Antonio, TX',
        #     'north_bound': 29.6,
        #     'south_bound': 29.4,
        #     'west_bound': -98.7,
        #     'east_bound': -98.5
        # },
        # 'dallas': {
        #     'name': 'Dallas, TX',
        #     'north_bound': 33.1,
        #     'south_bound': 32.8,
        #     'west_bound': -97.5,
        #     'east_bound': -96.9
        # },
        # 'las_vegas': {
        #     'name': 'Las Vegas, NV',
        #     'north_bound': 36.2,
        #     'south_bound': 35.9,
        #     'west_bound': -115.5,
        #     'east_bound': -115.0
        # },
        # 'phoenix': {
        #     'name': 'Phoenix, AZ',
        #     'north_bound': 33.7,
        #     'south_bound': 33.3,
        #     'west_bound': -112.2,
        #     'east_bound': -111.8
        # },
        # 'tucson': {
        #     'name': 'Tucson, AZ',
        #     'north_bound': 32.4,
        #     'south_bound': 32.1,
        #     'west_bound': -111.2,
        #     'east_bound': -110.8
        # },
        # 'edmonton': {
        #     'name': 'Edmonton, AB',
        #     'north_bound': 54.0,
        #     'south_bound': 53.5,
        #     'west_bound': -114.5,
        #     'east_bound': -113.5
        # },
        # 'winnipeg': {
        #     'name': 'Winnipeg, MB',
        #     'north_bound': 49.0,
        #     'south_bound': 48.5,
        #     'west_bound': -97.5,
        #     'east_bound': -96.5
        # },
        # 'calgary': {
        #     'name': 'Calgary, AB',
        #     'north_bound': 51.2,
        #     'south_bound': 50.8,
        #     'west_bound': -115.0,
        #     'east_bound': -113.5
        # },
        # 'fairview': {
        #     'name': 'Fairview, AB',
        #     'north_bound': 56.0,
        #     'south_bound': 55.5,
        #     'west_bound': -120.0,
        #     'east_bound': -119.5
        # },
        # 'grande_prairie': {
        #     'name': 'Grande Prairie, AB',
        #     'north_bound': 55.5,
        #     'south_bound': 55.0,
        #     'west_bound': -119.5,
        #     'east_bound': -119.0
        # },
        # 'peace_river': {
        #     'name': 'Peace River, AB',
        #     'north_bound': 56.0,
        #     'south_bound': 55.5,
        #     'west_bound': -118.5,
        #     'east_bound': -118.0
        # },
        'northwest_alberta': {
            'name': 'Northwest Alberta',
            'north_bound': 57.0,
            'south_bound': 54.5,
            'west_bound': -121.5,
            'east_bound': -116.0
        },
        'manitoba': {
            'name': 'Manitoba',
            'north_bound': 52.8,
            'south_bound': 49.0,
            'west_bound': -102.0,
            'east_bound': -95.0
        },
        'ab_sk_mb': {
            'name': 'Alberta, Saskatchewan, Manitoba',
            'north_bound': 60.0,
            'south_bound': 49.0,
            'west_bound': -121.5,
            'east_bound': -95.0
        },
        # 'portland': {
        #     'name': 'Portland, OR',
        #     'north_bound': 45.7,
        #     'south_bound': 45.4,
        #     'west_bound': -123.0,
        #     'east_bound': -122.5
        # },
        # 'gila_national_forest': {
        #     'name': 'Gila National Forest, NM',
        #     'north_bound': 34.0,
        #     'south_bound': 33,
        #     'west_bound': -109,
        #     'east_bound': -108.0
        # },
        # 'sioux_falls': {
        #     'name': 'Sioux Falls, SD',
        #     'north_bound': 43.7,
        #     'south_bound': 43.4,
        #     'west_bound': -97.5,
        #     'east_bound': -97.0
        # },
        # 'cheyenne': {
        #     'name': 'Cheyenne, WY',
        #     'north_bound': 41.2,
        #     'south_bound': 40.8,
        #     'west_bound': -105.5,
        #     'east_bound': -104.9
        # },
        'eastern_n_carolina': {
            'name': 'Eastern North Carolina',
            'north_bound': 36.0,
            'south_bound': 34.5,
            'west_bound': -78.0,
            'east_bound': -75.5
        },
        # 'new_mexico': {
        #     'name': 'New Mexico',
        #     'north_bound': 37.0,
        #     'south_bound': 32.0,
        #     'west_bound': -109,
        #     'east_bound': -103.0
        # },
        # 'southern_new_mexico': {
        #     'name': 'Southern New Mexico',
        #     'north_bound': 34.5,
        #     'south_bound': 31.0,
        #     'west_bound': -109.0,
        #     'east_bound': -103.0
        # },
        # 'southwest_texas': {
        #     'name': 'South West Texas',
        #     'north_bound': 32.0,
        #     'south_bound': 26.0,
        #     'west_bound': -106.5,
        #     'east_bound': -97.0
        # },
        # 'south_west_us': {
        #     'name': 'Southwest US',
        #     'north_bound': 35.0,
        #     'south_bound': 27.5,
        #     'west_bound': -110.0,
        #     'east_bound': -97.0
        # },
        # 'central_texas': {
        #     'name': 'Central Texas',
        #     'north_bound': 31.2,
        #     'south_bound': 30.6,
        #     'west_bound': -99.7,
        #     'east_bound': -98.3
        # },
        # 'dakotas_nebraska_minnesota': {
        #     'name': 'Midwest',
        #     'north_bound': 49.0,
        #     'south_bound': 42.0,
        #     'west_bound': -104.0,
        #     'east_bound': -90.0
        # },
        # 'wv_virginia_carolinas': {
        #     'name': 'West Virginia, Virginia, Carolinas',
        #     'north_bound': 39.0,
        #     'south_bound': 33.0,
        #     'west_bound': -85.0,
        #     'east_bound': -75.0
        # },
        # 'southern_california': {
        #     'name': 'Southern California',
        #     'north_bound': 35.0,
        #     'south_bound': 32.0,
        #     'west_bound': -120.0,
        #     'east_bound': -114.0
        # },
        # 'northern_california': {
        #     'name': 'Northern California',
        #     'north_bound': 42.0,
        #     'south_bound': 35.0,
        #     'west_bound': -125.0,
        #     'east_bound': -120.0
        # },
        # 'east_bay': {
        #     'name': 'East Bay, CA',
        #     'north_bound': 38.0,
        #     'south_bound': 36.0,
        #     'west_bound': -121.5,
        #     'east_bound': -119.5
        # },
        # 'record_location_6': {
        #     'name': 'Chihuahua, Mexico',
        #     'north_bound': 26.5,
        #     'south_bound': 25.5,
        #     'west_bound': -105.5,
        #     'east_bound': -104.5
        # },
        # 'chihuahua': {
        #     'name': 'Chihuahua, Mexico',
        #     'north_bound': 31.0,
        #     'south_bound': 25.0,
        #     'west_bound': -107.0,
        #     'east_bound': -103.0
        # },
        # 'sonora': {
        #     'name': 'Sonora, Mexico',
        #     'north_bound': 32.0,
        #     'south_bound': 25.0,
        #     'west_bound': -114.0,
        #     'east_bound': -108.0
        # },
        'az_nm_sonora': {
            'name': 'Arizona, New Mexico, Sonora',
            'north_bound': 32.0,
            'south_bound': 25.5,
            'west_bound': -111.0,
            'east_bound': -104.5
        },
        'drought_region': {
            'name': 'Southwest US & Northern Mexico',
            'north_bound': 32.5,
            'south_bound': 24.0,
            'west_bound': -111.0,
            'east_bound': -100.0
        },
        'northern_mexico': {
            'name': 'Northern Mexico',
            'north_bound': 32.0,
            'south_bound': 25.0,
            'west_bound': -115.0,
            'east_bound': -100.0
        },
        'missouri': {
            'name': 'Missouri',
            'north_bound': 40.5,
            'south_bound': 36.0,
            'west_bound': -95.0,
            'east_bound': -89.0
        },
        'midwest': {
            'name': 'Midwest US',
            'north_bound': 45.5,
            'south_bound': 38.0,
            'west_bound': -104.5,
            'east_bound': -93.0
        },
        'nebraska': {
            'name': 'Nebraska',
            'north_bound': 43.0,
            'south_bound': 40.0,
            'west_bound': -104.0,
            'east_bound': -94.0
        },
        # 'los_angeles': {
        #     'name': 'Los Angeles, CA',
        #     'north_bound': 34.5,
        #     'south_bound': 33.9,
        #     'west_bound': -118.7,
        #     'east_bound': -118.1
        # },
        # 'palm_springs': {
        #     'name': 'Palm Springs, CA',
        #     'north_bound': 33.9,
        #     'south_bound': 33.7,
        #     'west_bound': -116.6,
        #     'east_bound': -116.4
        # },
        # 'south_west': {
        #     'name': 'Southwest US',
        #     'north_bound': 37.0,
        #     'south_bound': 31.0,
        #     'west_bound': -115.0,
        #     'east_bound': -100.0
        # },

    }
    return locations

def retrieve_active_ca_fires(url='https://cwfis.cfs.nrcan.gc.ca/downloads/activefires/activefires.csv'):
    """
    Retrieve active fire data for Canada from the CWFIS website.
    
    Parameters:
    -----------
    url : str
        URL to the active fires CSV file on the CWFIS website
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing active fire data with columns:
        - 'lat'
        - 'lon'
        - 'hectares'
        - 'stage_of_control'
    """
    try:
        print(f"Retrieving active fire data from {url}...")
        df = pd.read_csv(url)
        print(f"Retrieved {len(df)} active fires.")
        df.columns = [col.strip() for col in df.columns]
        df = df[['lat', 'lon', 'hectares', 'stage_of_control']]
        return df
    except Exception as e:
        print(f"Error retrieving active fire data: {e}")
        return pd.DataFrame()    

def annotate_canada_map(input_map_path, active_fires_df, output_map_path=None):
    """
    Add active fires to an existing Canada FWI map.
    
    Parameters:
    -----------
    input_map_path : str
        Path to the existing Canada FWI map PNG file
    active_fires_df : pandas DataFrame
        DataFrame containing active fire data with columns:
        - 'lat': Latitude of the fire
        - 'lon': Longitude of the fire
        - 'hectares': Area burned in hectares
        - 'stage_of_control': Stage of control (e.g., 'OC', '0% contained', etc.)
    output_map_path : str, optional
        Path to save the annotated map (if None, appends "_annotated" to the input path)
    """
    # Define default output path if not specified
    if output_map_path is None:
        base_name = input_map_path.rsplit('.', 1)[0]
        output_map_path = f"{base_name}_annotated.png"
    
    print(f"Adding annotations to {input_map_path}")
    
    # Set up Space Mono font
    font_props = setup_space_mono_font()
    
    # Load the existing image as a matplotlib figure
    img = plt.imread(input_map_path)
    fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    
    # Set up a basemap style overlay for plotting points
    m_ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    m_ax.set_extent([-141, -52.6, 41.7, 70], crs=ccrs.PlateCarree())  # Canada extent
    m_ax.set_frame_on(False)  # Hide the frame
    m_ax.patch.set_alpha(0)   # Make the background transparent
    
    # Create consistent styling for fire markers
    marker_style = {
        'marker': '*',
        'markersize': 8,
        'markerfacecolor': 'red',
        'markeredgecolor': 'black',
        'markeredgewidth': 0.8,
        'alpha': 0.8,
        'zorder': 10,
        'transform': ccrs.PlateCarree()
    }
    
    # Add all fire markers
    if not active_fires_df.empty:
        m_ax.plot(active_fires_df['lon'], active_fires_df['lat'], 
                 linestyle='none', **marker_style)
        
        # Add legend on the left side of the map
        fire_count = len(active_fires_df)
        legend_text = f"{fire_count} Active Fire{'s' if fire_count != 1 else ''}"
        
        # Create a custom legend with the fire symbol
        legend_marker = plt.Line2D([0], [0], marker='*', color='w', 
                                 markerfacecolor='red', markeredgecolor='black',
                                 markersize=10, label=legend_text)
        
        # Position the legend in the upper left corner
        m_ax.legend(handles=[legend_marker], loc='upper left', 
                   fontsize=12, framealpha=0.7,
                   prop=font_props.get('bold') if font_props else None)
    else:
        print("No active fires to plot")
    
    # Save the annotated map
    plt.tight_layout(pad=0)
    plt.savefig(output_map_path, dpi=300, bbox_inches='tight')
    print(f"Saved annotated map to {output_map_path}")
    plt.close()

def generate_country_map(fwi_data, country_info, output_dir, start_date, end_date, days_back, 
                        suffix="raw", colorbar_label='Fire Weather Index Anomaly', 
                        transform_coords=False, annotate_fires=None, language='en'):
    """Generate FWI anomaly map for a single country with properly projected borders"""
    country_code = country_info['code']
    print(f"\nGenerating FWI anomaly map for {country_info['name']}...")
    
    # Setup Space Mono font properties first thing
    font_props = setup_space_mono_font()
    
    # Get coordinate names
    lat_name, lon_name = get_coord_names(fwi_data)
    
    # Download country shapefile
    try:
        country_gdf = get_country_gdf(country_code, output_dir=SHAPEFILE_DIR)
        print(f"Loaded shapefile with {len(country_gdf)} features for {country_info['name']}")
        
        # Use simplified geometry to reduce memory usage
        simplify_tolerance = 0.1 if country_code.lower() == 'can' else 0.05
        country_outline = country_gdf.dissolve().simplify(simplify_tolerance)
        print(f"Created simplified country outline")
    except Exception as e:
        print(f"Error loading shapefile for {country_info['name']}: {e}")
        return
    
    # Set up the figure with appropriate projection and apply consistent styling
    print("Setting up figure...")
    
    # Select the right projection based on country
    if country_code.lower() == 'can':
        # Use Lambert Conformal projection for Canada (better for northern latitudes)
        projection_type = "lambert"
        # Limit northern extent to improve aspect ratio
        country_info['bounds'][3] = min(country_info['bounds'][3], 70)
    else:
        # Use Albers Equal Area for US and Mexico
        projection_type = "albers"
    
    # Create map with selected projection
    fig = plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
    ax = setup_map_figure(projection_type=projection_type)
    
    # Add state boundaries only for USA if indicated
    if (country_info.get('add_states', False)):
        ax.add_feature(cfeature.STATES, linewidth=0.7, edgecolor='gray', alpha=0.8)
    
    # Extract regional data subset
    print(f"Setting map extent: {country_info['bounds'][0]} to {country_info['bounds'][1]} (lon), " +
          f"{country_info['bounds'][2]} to {country_info['bounds'][3]} (lat)")
    
    west = transform_longitude(country_info['bounds'][0]) if transform_coords else country_info['bounds'][0]
    east = transform_longitude(country_info['bounds'][1]) if transform_coords else country_info['bounds'][1]
    
    # Extract data for the region
    regional_fwi = fwi_data.sel(
        {lat_name: slice(country_info['bounds'][3], country_info['bounds'][2]),
         lon_name: slice(west, east)}
    )
    print(f"Regional data shape: {regional_fwi.shape}")
    
    # Create and apply country mask
    country_mask = create_country_mask(regional_fwi, country_gdf, transform_coords)
    masked_fwi = regional_fwi.where(country_mask)
    print(f"Applied country mask - data now strictly limited to {country_info['name']} boundaries")
    
    # Set map extent in DISPLAY projection coordinates
    try:
        ax.set_extent(
            [
                country_info['bounds'][0] - 1,  # Add padding
                country_info['bounds'][1] + 1,
                country_info['bounds'][2] - 1,
                country_info['bounds'][3] + 1
            ],
            crs=ccrs.PlateCarree()  # This is the coordinate system of the bounds
        )
    except Exception as e:
        print(f"Warning setting extent: {e}")
        # Alternative approach if the first fails
        ax.set_global()
        
    # Calculate regional statistics and plot data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        country_min = float(masked_fwi.min().values)
        country_max = float(masked_fwi.max().values)
    
    # Create a single consistent colormap for all plots
    custom_cmap = create_custom_colormap(anomaly_type=suffix)
    
    # Set appropriate normalization based on the anomaly type
    if suffix == "std":
        # Use more contrastive normalization for standard anomalies
        # Focus on -3 to +3 range (99.7% of normal distribution)
        norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        title_suffix = f"Range: {country_min:.1f}σ to {country_max:.1f}σ"
    elif suffix == "pct":
        # For percentage anomalies, use fixed bounds good for percentages
        norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=200)
        title_suffix = f"Range: {country_min:.1f}% to {country_max:.1f}%"
    else:
        # For raw anomalies, use dynamic range based on data
        country_vmax = max(abs(country_min), abs(country_max))
        norm = TwoSlopeNorm(vmin=-country_vmax, vcenter=0, vmax=country_vmax)
        title_suffix = f"Range: {country_min:.1f} to {country_max:.1f}"
    
    # Plot the data
    try:
        im = masked_fwi.plot(
            ax=ax,
            cmap=custom_cmap, 
            norm=norm,
            transform=ccrs.PlateCarree(),
            add_colorbar=False
        )
        print("Successfully plotted masked country data")
    except Exception as e:
        print(f"Error plotting data: {e}")
        plt.close()
        return
    
    # Add coastlines and borders for better reference
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray')
    
    # Add colorbar and labels with Space Mono font
    cbar = add_colorbar(ax, im, colorbar_label)
    print("Adding colorbar...")
    
    # Apply font properties to colorbar ticks and label
    if font_props:
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(font_props.get('regular'))
        cbar.set_label(colorbar_label, fontproperties=font_props.get('regular'), fontsize=14)
    
    # Use the format_plot_title function for consistent styling
    # First clear any existing matplotlib titles
    ax.set_title('')
    plt.suptitle('')
    
    # Add formatted title and subtitle using Space Mono with language support
    if language == 'fr':
        title = f'RISQUE D\'INCENDIE DE PRINTEMPS EXTRÊME À TRAVERS {country_info["title_prefix"]}'
        data_note = "DONNÉES : CEMS FIRE WEATHER INDEX. Moyenne FWI du 1er mars au 1er juin comparée à la ligne de base 1991-2020."
    else:
        title = f'EXTREME SPRING FIRE RISK ACROSS {country_info["title_prefix"]}'
        data_note = "DATA: CEMS FIRE WEATHER INDEX. March 1 - June 1 average FWI compared to 1991-2020 baseline."
    
    subtitle = ''
    format_plot_title(ax, title, subtitle, font_props)
    print("Adding titles and annotations...")
    
    # Add data source using Deep Sky branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    add_gridlines(ax, show_gridlines=False)
    
    # Save figures using the utility function
    base_output_file = f'{output_dir}/fwi_anomaly_{suffix}_map_{country_info["output_suffix"]}_{str(days_back)}_days'
    
    # Save as PNG and SVG
    save_plot(fig, base_output_file + '.png')
    print(f"Saving PNG to {base_output_file}.png...")
    
    # Save SVG separately since save_plot might modify the extension
    plt.savefig(f"{base_output_file}.svg", format='svg', bbox_inches='tight')
    print(f"Saving SVG to {base_output_file}.svg...")

     # Add fire annotations if requested for Canada maps
    if annotate_fires and country_code.lower() == 'can':
        print("Retrieving and adding active fire data to Canada map...")
        active_fires = retrieve_active_ca_fires()
        if not active_fires.empty:
            annotate_canada_map(f"{base_output_file}.png", active_fires)
    
    # Clean up
    plt.close()
    import gc
    gc.collect()
    print(f"Cleaning up memory...")
    print(f"Completed {country_info['name']} map")

def plot_record_cities_fwi(fwi_data, record_cities, locations_dict, start_date, end_date, 
                          days_back=30, output_file=None):
    """
    Create a line chart showing Fire Weather Index values for selected cities over multiple years.
    
    Parameters:
    -----------
    fwi_data : xarray.Dataset
        Dataset containing FWI data with time dimension and 'fwinx' variable
    record_cities : list
        List of city keys from locations_dict to include in the plot
    locations_dict : dict
        Dictionary containing location coordinates for each city
    start_date : datetime
        Start date of the current period
    end_date : datetime
        End date of the current period
    days_back : int
        Number of days to average for each data point
    output_file : str
        Path to save the output figure
    """
    if output_file is None:
        output_file = f'fwi_record_cities_comparison.png'
    
    print(f"\nGenerating multi-year line chart for {len(record_cities)} record cities...")
    
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Define years to analyze (last 10 years)
    current_year = end_date.year
    years_to_analyze = range(current_year - 9, current_year + 1)  # Last 10 years including current
    
    # Store data for each city
    city_data = {}
    
    # Define a colormap for consistent city colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(record_cities)))
    
    # Calculate FWI for each city across years
    for idx, city_key in enumerate(record_cities):
        if city_key not in locations_dict:
            print(f"Warning: City '{city_key}' not found in locations dictionary. Skipping.")
            continue
            
        coords = locations_dict[city_key]
        city_name = coords['name']
        print(f"Processing data for {city_name}...")
        
        # Get FWI values for each year
        yearly_fwi = []
        years_with_data = []
        
        for year in years_to_analyze:
            fwi_value = get_fwi_for_year(fwi_data, year, coords, end_date, days_back)
            
            if not np.isnan(fwi_value):
                yearly_fwi.append(fwi_value)
                years_with_data.append(year)
            else:
                print(f"  No data for {city_name} in {year}")
        
        # Store data for this city
        city_data[city_key] = {
            'name': city_name,
            'years': years_with_data,
            'fwi': yearly_fwi,
            'color': colors[idx]
        }
        
        # Plot the line for this city
        plt.plot(years_with_data, yearly_fwi, 'o-', color=colors[idx], linewidth=2, 
                label=f"{city_name}", markersize=8)
    
    # Highlight the current year
    plt.axvline(x=current_year, color='gray', linestyle='--', alpha=0.7, 
               label=f"Current Year ({current_year})")
    
    # Calculate month/day window for display
    month_day_start = start_date.strftime("%b %d")
    month_day_end = end_date.strftime("%b %d")
    
    # Add labels and title
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Fire Weather Index (FWI)', fontsize=12)
    plt.title(f'Fire Weather Index Trends for Selected Cities\n' +
             f'(Average FWI for {month_day_start} - {month_day_end} window each year)', 
             fontsize=14, fontweight='bold')
    
    # Add legend with better positioning
    plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    
    # Customize grid and axis
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set reasonable y-axis limits if we have data
    if any(city_data.values()):
        all_fwi_values = [fwi for city in city_data.values() for fwi in city['fwi']]
        if all_fwi_values:
            min_fwi = max(0, min(all_fwi_values) * 0.9)  # Ensure no negative values
            max_fwi = max(all_fwi_values) * 1.1  # Add some padding
            plt.ylim(min_fwi, max_fwi)
    
    # Add text annotations for highest values
    for city_key, data in city_data.items():
        if not data['fwi']:
            continue
            
        # Find the highest FWI value for this city
        max_idx = np.argmax(data['fwi'])
        max_year = data['years'][max_idx]
        max_fwi = data['fwi'][max_idx]
        
        # Only annotate if this is the highest value or it's the current year
        if max_year == max(data['years']) or max_year == current_year:
            plt.annotate(f"{max_fwi:.1f}",
                        (max_year, max_fwi),
                        xytext=(5, 5), textcoords='offset points',
                        color=data['color'],
                        fontweight='bold')
    
    # Show FWI value for the current year for each city
    text_lines = []
    for city_key, data in city_data.items():
        # Check if we have current year data
        if current_year in data['years']:
            current_year_idx = data['years'].index(current_year)
            current_fwi = data['fwi'][current_year_idx]
            text_lines.append(f"{data['name']}: {current_fwi:.1f}")
    
    if text_lines:
        plt.figtext(0.02, 0.02, f"Current year ({current_year}) FWI values:\n" + "\n".join(text_lines),
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add data source credit
    plt.figtext(0.99, 0.01, "Data source: CEMS Fire Weather Index", 
               ha='right', fontsize=8, style='italic')
    
    # Ensure x-axis shows all years with integer ticks
    plt.xticks(list(years_to_analyze))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-year comparison chart saved as '{output_file}'")
    plt.close()

def plot_country_fwi_anomaly_maps(fwi_anomaly, start_date, end_date, output_dir, days_back, 
                                suffix="raw", colorbar_label='Fire Weather Index Anomaly',
                                annotate_fires=None, language='en'):
    """Create separate FWI anomaly maps for Canada, contiguous USA, and Mexico with consistent coloring"""
    print(f"\nGenerating country-specific FWI anomaly maps in {language.upper()}...")
    
    # Get coordinate names 
    lat_name, lon_name = get_coord_names(fwi_anomaly)
    
    # DIAGNOSTIC: Check actual dataset ranges
    lon_min = fwi_anomaly[lon_name].min().values.item()
    lon_max = fwi_anomaly[lon_name].max().values.item()
    lat_min = fwi_anomaly[lat_name].min().values.item()
    lat_max = fwi_anomaly[lat_name].max().values.item()
    print(f"Dataset coordinate range: Longitude {lon_min:.2f} to {lon_max:.2f}, Latitude {lat_min:.2f} to {lat_max:.2f}")
    
    # Check coordinate system
    transform_coords = check_coordinate_system(fwi_anomaly, -180, 180)
    if transform_coords:
        print("Dataset uses 0-360 longitude system, will adjust country borders")
    else:
        print("Dataset uses -180 to 180 longitude system")
    
    # Define country configurations with language support
    if language == 'fr':
        countries = {
            'usa': {
                'name': 'États-Unis',
                'code': 'USA',
                'bounds': [-125, -66, 24, 49.5],
                'title_prefix': 'États-Unis',
                'output_suffix': 'usa_fr',
                'add_states': True
            },
            'canada': {
                'name': 'Canada',
                'code': 'CAN',
                'bounds': [-141, -52.6, 41.7, 83],
                'title_prefix': 'Canada',
                'output_suffix': 'canada_fr',
                'add_states': False
            },
            'mexico': {
                'name': 'Mexique',
                'code': 'MEX',
                'bounds': [-118.5, -86.5, 14.5, 32.5],
                'title_prefix': 'Mexique',
                'output_suffix': 'mexico_fr',
                'add_states': False
            }
        }
        # French colorbar label
        if 'Fire Weather Index Anomaly' in colorbar_label:
            colorbar_label = 'Anomalie de l\'Indice Météorologique d\'Incendie'
    else:
        countries = {
            'usa': {
                'name': 'United States',
                'code': 'USA',
                'bounds': [-125, -66, 24, 49.5],
                'title_prefix': 'U.S.',
                'output_suffix': 'usa',
                'add_states': True
            },
            'canada': {
                'name': 'Canada',
                'code': 'CAN',
                'bounds': [-141, -52.6, 41.7, 83],
                'title_prefix': 'Canada',
                'output_suffix': 'canada',
                'add_states': False
            },
            'mexico': {
                'name': 'Mexico',
                'code': 'MEX',
                'bounds': [-118.5, -86.5, 14.5, 32.5],
                'title_prefix': 'Mexico',
                'output_suffix': 'mexico',
                'add_states': False
            }
        }
    
    # Generate a map for each country
    for country_code, country_info in countries.items():
        generate_country_map(
            fwi_anomaly, 
            country_info, 
            output_dir, 
            start_date, 
            end_date, 
            days_back,
            suffix, 
            colorbar_label, 
            transform_coords,
            annotate_fires= annotate_fires if country_code == 'canada' else None,
            language=language
        )

def generate_combined_north_america_map(fwi_data, output_dir, start_date, end_date, days_back, 
                                      suffix="raw", colorbar_label='Fire Weather Index Anomaly', 
                                      transform_coords=False, annotate_fires=None, language='en'):
    """Generate FWI anomaly map combining US and Canada with properly projected borders"""
    print(f"\nGenerating combined North America FWI anomaly map...")
    
    # Setup Space Mono font properties first thing
    font_props = setup_space_mono_font()
    
    # Get coordinate names
    lat_name, lon_name = get_coord_names(fwi_data)
    
    # Define combined North America bounds (US + Canada)
    combined_bounds = [-141, -52.6, 24, 70]  # west, east, south, north
    
    # Download shapefiles for both countries
    try:
        usa_gdf = get_country_gdf('USA', output_dir=SHAPEFILE_DIR)
        canada_gdf = get_country_gdf('CAN', output_dir=SHAPEFILE_DIR)
        print(f"Loaded shapefiles for US ({len(usa_gdf)} features) and Canada ({len(canada_gdf)} features)")
        
        # Combine the country geometries
        combined_gdf = pd.concat([usa_gdf, canada_gdf], ignore_index=True)
        
        # Use simplified geometry to reduce memory usage
        simplify_tolerance = 0.05
        combined_outline = combined_gdf.dissolve().simplify(simplify_tolerance)
        print(f"Created simplified combined country outline")
    except Exception as e:
        print(f"Error loading shapefiles for combined map: {e}")
        return
    
    # Set up the figure with Lambert Conformal projection (good for North America)
    print("Setting up figure...")
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS['background'])
    ax = setup_map_figure(projection_type="lambert")
    
    # Add US state boundaries
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray', alpha=0.6)
    
    # Add Canadian provincial borders using Natural Earth data (same as explore_smoke.py)
    try:
        provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(provinces, linewidth=0.5, edgecolor='gray', alpha=0.6)
        print("Added Canadian provincial borders")
    except Exception as e:
        print(f"Could not load provincial borders: {e}")
    
    # Extract regional data subset
    print(f"Setting map extent: {combined_bounds[0]} to {combined_bounds[1]} (lon), " +
          f"{combined_bounds[2]} to {combined_bounds[3]} (lat)")
    
    west = transform_longitude(combined_bounds[0]) if transform_coords else combined_bounds[0]
    east = transform_longitude(combined_bounds[1]) if transform_coords else combined_bounds[1]
    
    # Extract data for the combined region
    regional_fwi = fwi_data.sel(
        {lat_name: slice(combined_bounds[3], combined_bounds[2]),
         lon_name: slice(west, east)}
    )
    print(f"Regional data shape: {regional_fwi.shape}")
    
    # Create and apply combined country mask
    combined_mask = create_country_mask(regional_fwi, combined_gdf, transform_coords)
    masked_fwi = regional_fwi.where(combined_mask)
    print(f"Applied combined country mask - data now limited to US and Canada boundaries")
    
    # map_extent = [-125, -60, 35, 60]  # Focused on smoke-affected regions 


    # Set map extent in DISPLAY projection coordinates
    try:
        ax.set_extent(
            [
                -130,  # West - include more of Alaska/western Canada
                -60,   # East - include Maritime provinces  
                35,    # South - include northern Mexico
                60     # North - focus on populated Canada, less Arctic
            ],
            crs=ccrs.PlateCarree()
        )
    except Exception as e:
        print(f"Warning setting extent: {e}")
        ax.set_global()
        
    # Calculate regional statistics and plot data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        region_min = float(masked_fwi.min().values)
        region_max = float(masked_fwi.max().values)
    
    # Create a single consistent colormap for all plots
    custom_cmap = create_custom_colormap(anomaly_type=suffix)
    
    # Set appropriate normalization based on the anomaly type
    if suffix == "std":
        norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        title_suffix = f"Range: {region_min:.1f}σ to {region_max:.1f}σ"
    elif suffix == "pct":
        pct_vmax = max(abs(region_min), abs(region_max), 100)  # At least ±100% range
        norm = TwoSlopeNorm(vmin=-pct_vmax, vcenter=0, vmax=pct_vmax)
        title_suffix = f"Range: {region_min:.1f}% to {region_max:.1f}%"
    else:
        region_vmax = max(abs(region_min), abs(region_max))
        norm = TwoSlopeNorm(vmin=-region_vmax, vcenter=0, vmax=region_vmax)
        title_suffix = f"Range: {region_min:.1f} to {region_max:.1f}"
    
    # Plot the data
    try:
        im = masked_fwi.plot(
            ax=ax,
            cmap=custom_cmap, 
            norm=norm,
            transform=ccrs.PlateCarree(),
            add_colorbar=False
        )
        print("Successfully plotted masked combined data")
    except Exception as e:
        print(f"Error plotting data: {e}")
        plt.close()
        return
    
    # Add coastlines and borders for better reference
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray')
    
    # Add colorbar and labels with Space Mono font
    cbar = add_colorbar(ax, im, colorbar_label)
    print("Adding colorbar...")
    
    # Apply font properties to colorbar ticks and label
    if font_props:
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(font_props.get('regular'))
        cbar.set_label(colorbar_label, fontproperties=font_props.get('regular'), fontsize=14)
    
    # Clear any existing matplotlib titles
    ax.set_title('')
    plt.suptitle('')
    
    # Add formatted title and subtitle using Space Mono with language support
    if language == 'fr':
        title = f'RISQUE D\'INCENDIE DE PRINTEMPS EXTRÊME EN AMÉRIQUE DU NORD'
        data_note = "DONNÉES : CEMS FIRE WEATHER INDEX. Moyenne FWI du 1er mars au 1er juin comparée à la ligne de base 1991-2020."
    else:
        title = f'EXTREME SPRING FIRE RISK ACROSS NORTH AMERICA'
        data_note = "DATA: CEMS FIRE WEATHER INDEX. March 1 - June 1 average FWI compared to 1991-2020 baseline."
    
    subtitle = ''
    format_plot_title(ax, title, subtitle, font_props)
    print("Adding titles and annotations...")
    
    # Add data source using Deep Sky branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    add_gridlines(ax, show_gridlines=False)
    
    # Save figures using the utility function
    base_output_file = f'{output_dir}/fwi_anomaly_{suffix}_map_north_america_{str(days_back)}_days'
    
    # Save as PNG and SVG
    save_plot(fig, base_output_file + '.png')
    print(f"Saving PNG to {base_output_file}.png...")
    
    # Save SVG separately
    plt.savefig(f"{base_output_file}.svg", format='svg', bbox_inches='tight')
    print(f"Saving SVG to {base_output_file}.svg...")

    # Add fire annotations if requested for Canada portion
    if annotate_fires:
        print("Retrieving and adding active fire data to North America map...")
        active_fires = retrieve_active_ca_fires()
        if not active_fires.empty:
            annotate_canada_map(f"{base_output_file}.png", active_fires)
    
    # Clean up
    plt.close()
    gc.collect()
    print(f"Cleaning up memory...")
    print(f"Completed North America map")

def plot_combined_north_america_fwi_anomaly_maps(fwi_anomaly_raw, fwi_anomaly_pct, fwi_anomaly_std, 
                                               start_date, end_date, output_dir, days_back, 
                                               annotate_fires=None, language='en'):
    """Create combined North America FWI anomaly maps for all three anomaly types"""
    print(f"\nGenerating combined North America FWI anomaly maps in {language.upper()}...")
    
    # Check coordinate system once for all maps
    lat_name, lon_name = get_coord_names(fwi_anomaly_raw)
    lon_min = fwi_anomaly_raw[lon_name].min().values.item()
    lon_max = fwi_anomaly_raw[lon_name].max().values.item()
    transform_coords = check_coordinate_system(fwi_anomaly_raw, -180, 180)
    
    if transform_coords:
        print("Dataset uses 0-360 longitude system, will adjust country borders")
    else:
        print("Dataset uses -180 to 180 longitude system")
    
    # Define anomaly types with their labels
    if language == 'fr':
        anomaly_configs = [
            {
                'data': fwi_anomaly_raw,
                'suffix': 'raw',
                'colorbar_label': 'Anomalie de l\'Indice Météorologique d\'Incendie'
            },
            {
                'data': fwi_anomaly_pct,
                'suffix': 'pct',
                'colorbar_label': 'Anomalie de l\'Indice Météorologique d\'Incendie (%)'
            },
            {
                'data': fwi_anomaly_std,
                'suffix': 'std',
                'colorbar_label': 'Anomalie Standardisée (σ)'
            }
        ]
    else:
        anomaly_configs = [
            {
                'data': fwi_anomaly_raw,
                'suffix': 'raw',
                'colorbar_label': 'Fire Weather Index Anomaly'
            },
            {
                'data': fwi_anomaly_pct,
                'suffix': 'pct',
                'colorbar_label': 'Fire Weather Index % Change'
            },
            {
                'data': fwi_anomaly_std,
                'suffix': 'std',
                'colorbar_label': 'Standardized Anomaly (σ)'
            }
        ]
    
    # Generate maps for each anomaly type
    for config in anomaly_configs:
        print(f"\nGenerating {config['suffix']} anomaly map...")
        generate_combined_north_america_map(
            config['data'],
            output_dir,
            start_date,
            end_date,
            days_back,
            suffix=config['suffix'],
            colorbar_label=config['colorbar_label'],
            transform_coords=transform_coords,
            annotate_fires=annotate_fires if config['suffix'] == 'raw' else None,
            language=language
        )
        
def record_fwi_checker(fwi_data, year_range=(2010, 2025), grid_resolution=1.0, min_years_required=8, country_bounds=None):
    """
    Systematically search for locations where the current year (2025) has the highest FWI value 
    in the given year range.
    """
    print(f"\nSearching for locations with record FWI values in {year_range[1]}...")
    available_years = sorted(set(fwi_data.valid_time.dt.year.values))
    print(f"Available years in dataset: {available_years}")
    missing_years = [year for year in range(year_range[0], year_range[1] + 1) 
                     if year not in available_years]
    if missing_years:
        print(f"WARNING: Some years in range {year_range} are not in the dataset: {missing_years}")
    
    # Try with a specific test location first (e.g., Albuquerque, NM)
    test_lat, test_lon = 35.0, -106.5  # Albuquerque coordinates
    if transform_coords:
        test_lon = transform_longitude(test_lon)
        
    print(f"\nTesting with known location: Albuquerque, NM ({test_lat}, {test_lon if not transform_coords else -106.5})")
    test_years_data = {}
    
    # Get most recent date in the dataset
    end_date = pd.to_datetime(fwi_data.valid_time.max().values)
    days_back = 90  # Same as in main function
    
    # Test with different selection methods
    for year in range(year_range[0], year_range[1] + 1):
        year_end_date = datetime.datetime(year, end_date.month, end_date.day)
        year_start_date = year_end_date - pd.Timedelta(days=days_back)
        
        # Create a time mask for the specific year and date range
        time_mask = (
            (fwi_data.valid_time >= np.datetime64(year_start_date)) & 
            (fwi_data.valid_time <= np.datetime64(year_end_date))
        )
        
        # Select data for this time period
        year_data = fwi_data.sel(valid_time=time_mask)
        
        if year_data.sizes['valid_time'] == 0:
            print(f"  No data for time period in {year}")
            continue
            
        # Find the closest grid points
        try:
            # Method 1: Using nearest neighbor for both lat and lon
            point_data = year_data.sel({
                lat_name: test_lat, 
                lon_name: test_lon
            }, method='nearest')
            
            mean_fwi = point_data['fwinx'].mean(dim='valid_time').values.item()
            test_years_data[year] = mean_fwi
            print(f"  {year} FWI (nearest): {mean_fwi:.2f}")
            
            # Method 2: Using a slice to select a region
            region_data = year_data.sel({
                lat_name: slice(test_lat + 1, test_lat - 1),
                lon_name: slice(test_lon - 1, test_lon + 1)  
            })
            
            if not region_data['fwinx'].isnull().all():
                region_mean = region_data['fwinx'].mean(dim=[lat_name, lon_name]).mean(dim='valid_time').values.item()
                print(f"  {year} FWI (region): {region_mean:.2f}")
            else:
                print(f"  {year} FWI (region): All NaN")
                
        except Exception as e:
            print(f"  Error for {year}: {e}")
    
    # Create actual search grid - using proper coordinates based on the test results
    print("\nCreating search grid with proper coordinates...")
    
    # Create a grid of latitudes in the correct range
    lats = np.arange(miny, maxy, grid_resolution)
    
    # For longitudes, ensure they're in the correct system for the dataset
    if transform_coords and search_minx > search_maxx:
        # Handle the case where min > max after transformation (crossing the date line)
        part1 = np.arange(search_minx, 360, grid_resolution)
        part2 = np.arange(0, search_maxx, grid_resolution)
        lons = np.concatenate([part1, part2])
    else:
        lons = np.arange(search_minx, search_maxx, grid_resolution)
    
    print(f"Generated search grid with {len(lons)}×{len(lats)} = {len(lons)*len(lats)} points")
    print(f"Grid spans longitude: {minx}° to {maxx}° and latitude: {miny}° to {maxy}°")
    print(f"Data longitude range: {fwi_data[lon_name].min().values.item():.1f} to {fwi_data[lon_name].max().values.item():.1f}")
    print(f"Data latitude range: {fwi_data[lat_name].min().values.item():.1f} to {fwi_data[lat_name].max().values.item():.1f}")
    
    # Prepare to store results
    record_locations = []
    
    # Counter for progress tracking
    total_points = len(lons) * len(lats)
    points_checked = 0
    last_percentage = -1
    nan_count = 0
    valid_point_count = 0
    
    print("Analyzing grid points...")
    
    # Iterate through each grid point
    for lon in lons:
        for lat in lats:
            # Update progress
            points_checked += 1
            percentage = (points_checked / total_points) * 100
            if int(percentage) > last_percentage:
                print(f"Progress: {int(percentage)}% complete ({valid_point_count} valid points found)", end="\r")
                last_percentage = int(percentage)
            
            # Get yearly FWI values for this location using nearest-neighbor selection
            yearly_fwi = {}
            valid_years = 0
            has_nan = False
            
            # Check each year
            for year in range(year_range[0], year_range[1] + 1):
                year_end_date = datetime.datetime(year, end_date.month, end_date.day)
                year_start_date = year_end_date - pd.Timedelta(days=days_back)
                
                # Create a time mask for the specific year and date range
                time_mask = (
                    (fwi_data.valid_time >= np.datetime64(year_start_date)) & 
                    (fwi_data.valid_time <= np.datetime64(year_end_date))
                )
                
                # Select data for this time period
                year_data = fwi_data.sel(valid_time=time_mask)
                
                if year_data.sizes['valid_time'] == 0:
                    has_nan = True
                    continue
                    
                # Use nearest-neighbor to find the closest grid point
                try:
                    point_data = year_data.sel({
                        lat_name: lat, 
                        lon_name: lon
                    }, method='nearest')
                    
                    if not point_data['fwinx'].isnull().all():
                        mean_fwi = point_data['fwinx'].mean(dim='valid_time').values.item()
                        yearly_fwi[year] = mean_fwi
                        valid_years += 1
                    else:
                        has_nan = True
                except Exception as e:
                    has_nan = True
            
            if has_nan and not yearly_fwi:
                nan_count += 1
                continue
                
            # If we found at least one valid data point, count it
            if yearly_fwi:
                valid_point_count += 1
            
            # Skip if we don't have enough years of data or current year is missing
            if valid_years < min_years_required or current_year not in yearly_fwi:
                continue
            
            # Check if current year is the highest
            current_year_fwi = yearly_fwi[current_year]
            previous_max_fwi = max([v for k, v in yearly_fwi.items() if k != current_year], default=0)
            
            # Calculate percentage increase over previous max
            if previous_max_fwi > 0 and current_year_fwi > previous_max_fwi:
                percent_increase = ((current_year_fwi / previous_max_fwi) - 1) * 100
                
                # Store the record location - convert back to -180 to 180 system for display
                display_lon = lon if not transform_coords else (lon - 360 if lon > 180 else lon)
                
                record_locations.append({
                    'longitude': display_lon,
                    'latitude': lat,
                    'current_fwi': current_year_fwi,
                    'previous_max_fwi': previous_max_fwi,
                    'percent_increase': percent_increase,
                    'years_with_data': valid_years,
                    'values_by_year': yearly_fwi
                })
    
    print(f"\nProcessed {total_points} grid points, found {valid_point_count} with valid data and {nan_count} with incomplete data")
    print(f"Found {len(record_locations)} locations with record high FWI values in {current_year}")
    
    # Convert to DataFrame
    if record_locations:
        results_df = pd.DataFrame(record_locations)
        
        # Sort by percent increase
        results_df = results_df.sort_values('percent_increase', ascending=False)
        
        # Print top results
        print("\nTop locations with record FWI values:")
        for i, row in results_df.head(16).iterrows():
            display_lon = row['longitude']
            years_str = ", ".join([f"{year}: {row['values_by_year'].get(year, 'N/A'):.1f}" 
                                  for year in range(year_range[0], year_range[1] + 1)])
            
            print(f"Location {i+1}: ({display_lon:.2f}, {row['latitude']:.2f}) - "
                 f"Current: {row['current_fwi']:.1f}, Previous Max: {row['previous_max_fwi']:.1f}, "
                 f"Increase: {row['percent_increase']:.1f}%")
            print(f"  Year values: {years_str}")
            
        # Generate suggested location entries for define_location_coords()
        print("\nSuggested entries for define_location_coords():")
        for i, row in results_df.head(5).iterrows():
            lon = row['longitude']
            lat = row['latitude']
            location_key = f"record_location_{i+1}"
            
            print(f"'{location_key}': {{")
            print(f"    'name': 'Record FWI Location {i+1}',")
            print(f"    'north_bound': {lat + 0.5},")
            print(f"    'south_bound': {lat - 0.5},")
            print(f"    'west_bound': {lon - 0.5},")
            print(f"    'east_bound': {lon + 0.5}")
            print("},")
        
        return results_df
    else:
        print("No locations found with record high FWI values in the current year.")
        return pd.DataFrame()
    # Plot FWI anomaly maps for each location
def add_mini_map(fig, coords, position=(0.75, 0.2), width=0.2, height=0.2, padding=9.0):
    """
    Add a miniature map in the corner of a figure showing the location of the data coordinates.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to add the minimap to
    coords : dict
        Dictionary with keys 'north_bound', 'south_bound', 'west_bound', 'east_bound'
    position : tuple
        Position of the lower left corner of the minimap as (x, y) in figure coordinates
    width : float
        Width of the minimap in figure coordinates
    height : float
        Height of the minimap in figure coordinates
    padding : float
        Padding around the location rectangle in degrees
        
    Returns:
    --------
    matplotlib.figure.Figure
        The input figure with minimap added
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Extract coordinates with padding for context
    west = coords['west_bound'] - padding
    east = coords['east_bound'] + padding
    south = coords['south_bound'] - padding
    north = coords['north_bound'] + padding
    
    # Calculate center coordinates for the area of interest
    center_lon = (coords['west_bound'] + coords['east_bound']) / 2
    center_lat = (coords['north_bound'] + coords['south_bound']) / 2
    
    # Create a new axis for the minimap with cartopy projection
    ax_mini = fig.add_axes([position[0], position[1], width, height], 
                         projection=ccrs.PlateCarree(),
                         facecolor='lightgray')
    
    # Add basic geographic features
    ax_mini.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax_mini.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax_mini.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
    ax_mini.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_mini.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':', alpha=0.7)
    ax_mini.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Set the map extent with padding for context
    ax_mini.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    
    # Draw a rectangle for the actual coordinates
    rect = plt.Rectangle(
        (coords['west_bound'], coords['south_bound']),
        coords['east_bound'] - coords['west_bound'],
        coords['north_bound'] - coords['south_bound'],
        linewidth=1.5,
        edgecolor='red',
        facecolor='none',
        alpha=0.8,
        transform=ccrs.PlateCarree()
    )
    ax_mini.add_patch(rect)
    
    # Add a small dot at the center of the area
    # ax_mini.plot(center_lon, center_lat, 'ro', markersize=3, transform=ccrs.PlateCarree())
    
    # Remove axes ticks and labels for cleaner appearance
    ax_mini.set_xticks([])
    ax_mini.set_yticks([])
    ax_mini.set_frame_on(True)
    
    # Add a title with the location name
    # ax_mini.set_title(coords['name'], fontsize=8)
    
    return fig

def create_fwi_line_chart_french(fwi_data, coords, end_date, days_back=90, output_file=None, include_minimap=True):
    """
    Create a French version of the FWI line chart showing values across years for a specific area
    """
    if output_file is None:
        output_file = f'{coords["name"].replace(" ", "_")}_fwi_line_by_year_fr.png'
    
    # Get current year
    current_year = end_date.year
    
    # Define years to analyze (2015-2025 period)
    years_to_analyze = range(2015, current_year + 1)

    # Calculate mean FWI for each year (reuse existing function)
    year_fwi_values = []
    valid_years = []
    
    for year in years_to_analyze:
        year_fwi = get_fwi_for_year(fwi_data, year, coords, end_date, days_back)
        if not np.isnan(year_fwi):
            year_fwi_values.append(year_fwi)
            valid_years.append(year)
        else:
            print(f"No valid data for {coords['name']} in {year}")

    if valid_years:
        # Create line chart
        fig, ax, font_props = setup_enhanced_plot(figsize=(14, 8))
        
        # Plot line with markers
        plt.plot(valid_years, year_fwi_values, 'o-', 
                 color=COLORS['comparison'], 
                 linewidth=2.5, 
                 markersize=8)
        
        # Highlight the most recent year with a different color
        if valid_years and valid_years[-1] == current_year:
            plt.plot(valid_years[-1], year_fwi_values[-1], 'o', 
                    color=COLORS['primary'], 
                    markersize=10)
            
            # Add annotation for current year
            plt.annotate(f"{year_fwi_values[-1]:.1f}",
                        (valid_years[-1], year_fwi_values[-1]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        color=COLORS['primary'],
                        fontweight='bold',
                        fontproperties=font_props.get('bold'))
        
        # French translations for location names
        location_translations = {
            'Southwest US & Northern Mexico': 'Sud-Ouest des États-Unis et Nord du Mexique',
            'Northwest Alberta': 'Nord-Ouest de l\'Alberta',
            'Alberta, Saskatchewan, Manitoba': 'Alberta, Saskatchewan, Manitoba',
            'Manitoba': 'Manitoba',
            'Eastern North Carolina': 'Caroline du Nord de l\'Est',
            'Arizona, New Mexico, Sonora': 'Arizona, Nouveau-Mexique, Sonora',
            'Northern Mexico': 'Nord du Mexique',
            'Missouri': 'Missouri',
            'Midwest US': 'Midwest des États-Unis',
            'Nebraska': 'Nebraska'
        }
        
        # Get French location name
        french_location = location_translations.get(coords['name'], coords['name'])
        
        # Format plot title with French text
        title = f"LE RISQUE D'INCENDIE DANS {french_location.upper()}\n A ATTEINT DES NIVEAUX RECORDS"
        subtitle = "INDICE MÉTÉOROLOGIQUE D'INCENDIE MOYEN DE PRINTEMPS"
        format_plot_title(ax, title, subtitle, font_props)
        
        # Format x-axis to show all years
        plt.xticks(list(years_to_analyze))
        plt.xlim(min(years_to_analyze) - 0.5, max(years_to_analyze) + 0.5)
        
        # Apply font properties to ticks
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_props.get('regular'))
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_props.get('regular'))
            
        # Add Deep Sky branding with French citation
        add_deep_sky_branding(ax, font_props, data_note=f"DONNÉES : CEMS FIRE WEATHER INDEX. Moyenne FWI du 1er mars au 1er juin.")
        
        # Add minimap if requested
        if include_minimap:
            fig = add_mini_map(fig, coords)
        
        # Save plot with consistent format
        save_plot(fig, output_file)
        print(f"\nFrench line chart saved as '{output_file}'")
        return fig
    else:
        print(f"\nNo valid data found for {coords['name']} area across years. Cannot create French chart.")
        return None


#######################
# MAIN FUNCTION
#######################

def main():
    """Main execution function"""
    # Set constants
    DATA_DIR = '../data/wildfires/'
    FILE_PATTERN = 'CEMS_fwi_north_america_*.nc'
    DAYS_BACK = 90
    OUTPUT_DIR = 'figures'
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)

    locations = define_location_coords()
    record_cities = ['tucson', 'santa_fe', 'el_paso', 'albuquerque']

    # Find and load data
    matching_files = find_nc_files(DATA_DIR, FILE_PATTERN)
    print(f"Found {len(matching_files)} matching files")
    
    if not matching_files:
        print("No files found. Please check the DATA_DIR and FILE_PATTERN.")
        return
    
    # Load FWI data
    fwi_data = load_nc_fwi_files(matching_files)

    
    # Calculate all three anomaly types
    fwi_anomaly_raw, fwi_anomaly_pct, fwi_anomaly_std, start_date, end_date, _, _ = calculate_fwi_anomaly(
        fwi_data, days_back=DAYS_BACK, return_all=True
    )

    # record_locations = record_fwi_checker(
    #     fwi_data, 
    #     year_range=(2010, 2025),
    #     grid_resolution=3.0,  # Use 2-degree resolution for faster initial search
    #     country_bounds=(-125, 24, -66, 49.5)  # Continental US
    # )
    
    plot_country_fwi_anomaly_maps(
        fwi_anomaly_raw,
        start_date,
        end_date,
        OUTPUT_DIR,
        DAYS_BACK,
        suffix="raw",
        colorbar_label="Fire Weather Index Anomaly",
        annotate_fires=True,
        language='en'
    )

    
    # # # Create maps for percentage anomaly
    # # plot_country_fwi_anomaly_maps(
    # #     fwi_anomaly_pct,
    # #     start_date,
    # #     end_date,
    # #     OUTPUT_DIR,
    # #     DAYS_BACK,
    # #     suffix="pct",
    # #     colorbar_label="Fire Weather Index % Change"
    # # )
    
    # # # Create maps for standardized anomaly
    # # plot_country_fwi_anomaly_maps(
    # #     fwi_anomaly_std,
    # #     start_date,
    # #     end_date,
    # #     OUTPUT_DIR,
    # #     DAYS_BACK,
    # #     suffix="std",
    # #     colorbar_label="Standardized Anomaly (σ)"
    # # )

    # Define locations for city-specific analysis
    # for city, coords in locations.items():
    #     print(f"\nGenerating line chart for {coords['name']}")
        
    #     # Create line chart for each city
    #     create_fwi_line_chart(
    #         fwi_data,
    #         coords,
    #         end_date,
    #         days_back=DAYS_BACK,
    #         output_file=f"{OUTPUT_DIR}/location_line_charts/{city}_fwi_line_by_year.png"
    #     )
    
    # =====================
    # FRENCH LANGUAGE CHARTS
    # =====================
    # print("\nGenerating French language FWI charts...")

    # # Generate French anomaly maps
    # plot_country_fwi_anomaly_maps(
    #     fwi_anomaly_raw,
    #     start_date,
    #     end_date,
    #     OUTPUT_DIR,
    #     DAYS_BACK,
    #     suffix="raw",
    #     colorbar_label="Anomalie de l'Indice Météorologique d'Incendie",
    #     annotate_fires=True,
    #     language='fr'
    # )

    # # Generate French location charts
    # location_coords = define_location_coords()
    # for location_key, coords in location_coords.items():
    #     print(f"\nProcessing French FWI chart for {coords['name']}...")
        
    #     french_output_file = f'figures/{location_key}_fwi_line_by_year_fr.png'
    #     fig_french = create_fwi_line_chart_french(
    #         fwi_data, 
    #         coords, 
    #         end_date, 
    #         days_back=DAYS_BACK, 
    #         output_file=french_output_file,
    #         include_minimap=True
    #     )

    plot_combined_north_america_fwi_anomaly_maps(
        fwi_anomaly_raw,
        fwi_anomaly_pct, 
        fwi_anomaly_std,
        start_date,
        end_date,
        OUTPUT_DIR,
        DAYS_BACK,
        annotate_fires=False,
        language='en'
    )



if __name__ == "__main__":
    main()



