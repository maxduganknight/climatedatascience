import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
from datetime import datetime, timedelta
import warnings
import os  # Add this missing import

from utils import (
    setup_space_mono_font, setup_enhanced_plot, format_plot_title,
    add_deep_sky_branding, save_plot, COLORS
)

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_smoke_data(nc_file_path):
    """Load and process the smoke dispersion data"""
    print("Loading smoke data...")
    ds = xr.open_dataset(nc_file_path)
    print(f"Dataset loaded with dimensions: {ds.dims}")
    return ds

def decode_time_flags(tflag):
    """Convert TFLAG format (YYYYDDD,HHMMSS) to datetime objects"""
    print("Decoding time information...")
    
    # Extract date and time components
    dates = tflag[:, 0, 0].values  # YYYYDDD format
    times = tflag[:, 0, 1].values  # HHMMSS format
    
    datetime_list = []
    for date_val, time_val in zip(dates, times):
        # Convert numpy integers to Python integers
        date_val = int(date_val)
        time_val = int(time_val)
        
        # Parse YYYYDDD format
        year = date_val // 1000
        day_of_year = date_val % 1000
        
        # Parse HHMMSS format
        hours = time_val // 10000
        minutes = (time_val % 10000) // 100
        seconds = time_val % 100
        
        # Create datetime
        base_date = datetime(year, 1, 1) + timedelta(days=day_of_year-1)
        full_datetime = base_date.replace(hour=hours, minute=minutes, second=seconds)
        datetime_list.append(full_datetime)
    
    return datetime_list

def check_coordinate_system(smoke_data, bbox_minx, bbox_maxx, lons=None):
    """Check for coordinate system mismatch and return transform flag"""
    if smoke_data is None and lons is None:
        # If no data provided, assume no transformation needed
        return False
    
    # Get coordinate names
    lat_name, lon_name = get_coord_names(smoke_data)
    
    if hasattr(smoke_data, lon_name):
        lon_min_smoke = smoke_data[lon_name].min().values.item()
        lon_max_smoke = smoke_data[lon_name].max().values.item()
    elif lons is not None:
        lon_min_smoke = lons.min()
        lon_max_smoke = lons.max()
    else:
        # No coordinate data available, assume no transformation needed
        return False
    
    # If smoke data is in 0-360 longitude and map expects -180 to 180
    transform_coords = lon_min_smoke >= 0 and lon_max_smoke > 180 and bbox_minx < 0
    return transform_coords

def get_coord_names(dataset_or_coords):
    """Extract coordinate names from dataset or use standard names for coordinate arrays"""
    if hasattr(dataset_or_coords, 'dims'):
        # This is an xarray dataset
        lat_name = 'latitude' if 'latitude' in dataset_or_coords.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in dataset_or_coords.dims else 'lon'
        return lat_name, lon_name
    else:
        # This is coordinate arrays, return standard names
        return 'lat', 'lon'

def transform_longitude(lon_value, transform=True):
    """Transform longitude between -180:180 and 0:360 systems"""
    if transform and lon_value > 180:
        return lon_value - 360
    elif transform and lon_value < 0:
        return lon_value + 360
    return lon_value

def estimate_coordinates(ds):
    """Estimate lat/lon coordinates from grid attributes - corrected for actual coordinate system"""
    print("Estimating geographic coordinates...")
    
    try:
        # Get grid information from attributes
        xcell = ds.attrs.get('XCELL', 0.1)  # Grid cell size 
        ycell = ds.attrs.get('YCELL', 0.1)
        xorig = ds.attrs.get('XORIG', -160.0)  # Grid origin longitude
        yorig = ds.attrs.get('YORIG', 32.0)    # Grid origin latitude
        
        print(f"Grid cell size: {xcell}° x {ycell}°")
        print(f"Grid origin: ({xorig}°, {yorig}°)")
        
        # Get grid dimensions
        ncols = ds.sizes['COL']
        nrows = ds.sizes['ROW']
        
        # The grid attributes indicate this is actually a lat/lon grid in degrees
        # Create coordinate arrays directly
        lons = xorig + np.arange(ncols) * xcell
        lats = yorig + np.arange(nrows) * ycell
        
        print(f"Calculated coordinate ranges:")
        print(f"  Longitude: {lons.min():.2f}° to {lons.max():.2f}°")
        print(f"  Latitude: {lats.min():.2f}° to {lats.max():.2f}°")
        
        # Check if this covers the expected North American domain
        expected_west = -170  # Alaska
        expected_east = -50   # Atlantic Canada
        expected_south = 20   # Southern Mexico/US
        expected_north = 75   # Arctic Canada
        
        if (lons.min() >= expected_west and lons.max() <= expected_east and 
            lats.min() >= expected_south and lats.max() <= expected_north):
            print("✓ Coordinate ranges look correct for North American domain")
        else:
            print("⚠ Warning: Coordinate ranges may not cover expected North American domain")
        
        return lons, lats
        
    except Exception as e:
        print(f"Error estimating coordinates: {e}")
        # Fallback - but this shouldn't be needed now
        ncols = ds.sizes['COL']
        nrows = ds.sizes['ROW']
        lons = np.linspace(-170, -50, ncols)  # Broader range including Alaska
        lats = np.linspace(20, 75, nrows)     # Broader range including Arctic
        
        print(f"Using fallback coordinates: Lon {lons.min():.2f} to {lons.max():.2f}, Lat {lats.min():.2f} to {lats.max():.2f}")
        return lons, lats

def create_smoke_colormap():
    """Create a colormap for smoke concentration visualization (CBC style with more red)"""
    # Define smoke concentration colors with more red tones
    colors_list = [
        '#ffffff00',  # Good (0-12 μg/m³) - Transparent (no color)
        '#e6d2b5',    # Moderate (12-35 μg/m³) - Very light tan
        '#d4a574',    # Unhealthy for Sensitive Groups (35-55 μg/m³) - Light reddish-brown
        '#c67c3b',    # Unhealthy (55-150 μg/m³) - Medium reddish-brown  
        '#a0522d',    # Very Unhealthy (150-250 μg/m³) - Dark reddish-brown
        '#8b0000',    # Hazardous (250+ μg/m³) - Dark red
    ]
    
    # Define boundaries based on air quality standards
    boundaries = [0, 12, 35, 55, 150, 250, 500]
    
    cmap = colors.ListedColormap(colors_list)
    norm = BoundaryNorm(boundaries, cmap.N)
    
    return cmap, norm, boundaries

def process_smoke_data(ds):
    """Process the smoke data for visualization"""
    print("Processing smoke data...")
    
    # Get PM2.5 data
    pm25 = ds['PM25']
    
    # Decode time information
    datetime_list = decode_time_flags(ds['TFLAG'])
    
    # Add time coordinate
    pm25_with_time = pm25.assign_coords(time=('TSTEP', datetime_list))
    
    # Find June 3rd at 12pm EST (17:00 UTC) or closest available
    target_datetime = datetime(2025, 6, 3, 17, 0)  # 12pm EST = 17:00 UTC
    closest_idx = 0
    min_diff = float('inf')
    
    for i, dt in enumerate(datetime_list):
        diff = abs((dt - target_datetime).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    
    # Get the closest data to June 3rd 12pm EST
    latest_pm25 = pm25_with_time.isel(TSTEP=closest_idx, LAY=0) # Surface layer
    actual_timestamp = datetime_list[closest_idx]
    
    print(f"Target datetime: {target_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Actual data timestamp: {actual_timestamp}")
    print(f"PM2.5 data range: {latest_pm25.min().values:.2f} - {latest_pm25.max().values:.2f} μg/m³")
    
    return latest_pm25, actual_timestamp

def plot_smoke_map(pm25_data, timestamp, lons, lats, output_file='smoke_map_north_america.png', language='en'):
    """Create a map visualization of smoke concentrations (enhanced style)"""
    print(f"Creating smoke concentration map in {language.upper()}...")
    
    # Debug: Print coordinate information
    print(f"Input coordinate ranges:")
    print(f"  Longitude: {lons.min():.2f}° to {lons.max():.2f}°")
    print(f"  Latitude: {lats.min():.2f}° to {lats.max():.2f}°")
    print(f"  Data shape: {pm25_data.shape}")
    
    # Set up the map figure with minimal whitespace
    fig = plt.figure(figsize=(16, 12))
    
    # Use Lambert Conformal Conic projection for North America
    proj = ccrs.LambertConformal(central_longitude=-100, central_latitude=50)
    
    # Create axes that fill most of the figure, leaving space only for title/subtitle and colorbar
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.75], projection=proj)  # [left, bottom, width, height]
    
    # Zoom in to focus on smoke-affected areas (Northern and Eastern North America)
    map_extent = [-125, -60, 35, 60]  # Focused on smoke-affected regions 
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    # Add map features with refined styling
    ax.add_feature(cfeature.LAND, facecolor='#f8f8f8', alpha=0.9)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff', alpha=0.7)
    ax.add_feature(cfeature.LAKES, facecolor='#b8d4f0', alpha=0.6)
    
    # Add international borders (more prominent)
    ax.add_feature(cfeature.BORDERS, linewidth=1.2, edgecolor='#666666', alpha=0.8)
    
    # Add US state borders
    ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='#999999', alpha=0.6)
    
    # Add Canadian province borders using Natural Earth data
    try:
        provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(provinces, linewidth=0.6, edgecolor='#999999', alpha=0.6)
    except Exception as e:
        print(f"Could not load province borders: {e}")
    
    # Create enhanced smoke colormap
    cmap, norm, boundaries = create_smoke_colormap()
    
    # Create coordinate meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Mask out values below 12 μg/m³ and apply smoothing
    masked_data = np.ma.masked_where(pm25_data.values < 12, pm25_data.values)
    
    # Apply light smoothing for better visual appeal
    try:
        from scipy.ndimage import gaussian_filter
        smoothed_data = gaussian_filter(masked_data.filled(0), sigma=0.8)
        smoothed_data = np.ma.masked_where(smoothed_data < 12, smoothed_data)
    except ImportError:
        print("scipy not available, using original data without smoothing")
        smoothed_data = masked_data
    except Exception as e:
        print(f"Smoothing failed: {e}, using original data")
        smoothed_data = masked_data
    
    # Plot the smoke data with enhanced styling
    im = ax.pcolormesh(lon_grid, lat_grid, smoothed_data, 
                       cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), 
                       shading='gouraud', alpha=0.85)
    
    # Enhanced colorbar positioned below the map
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    
    # Set colorbar label based on language
    if language == 'fr':
        cbar_label = 'Concentration de PM2.5 (μg/m³)'
        # French AQI category labels
        aqi_labels = ['Modéré\n(12-35)', 'Malsain pour les\nGroupes Sensibles\n(35-55)', 
                      'Malsain\n(55-150)', 'Très Malsain\n(150-250)', 'Dangereux\n(250+)']
    else:
        cbar_label = 'PM2.5 Concentration (μg/m³)'
        # English AQI category labels
        aqi_labels = ['Moderate\n(12-35)', 'Unhealthy for\nSensitive Groups\n(35-55)', 
                      'Unhealthy\n(55-150)', 'Very Unhealthy\n(150-250)', 'Hazardous\n(250+)']
    
    cbar.set_label(cbar_label, fontsize=12, weight='bold', color='#333333')
    
    # Set colorbar ticks and labels
    tick_positions = [23.5, 45, 102.5, 200, 375]  # Centered positions
    cbar.ax.set_xticks(tick_positions)
    cbar.ax.set_xticklabels(aqi_labels, rotation=0, ha='center', fontsize=9, 
                           color='#333333')
    
    # Style the colorbar
    cbar.ax.tick_params(size=0)  # Remove tick marks
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#cccccc')
    
    # Enhanced title with EST time and language support
    est_time = timestamp.replace(tzinfo=None) - timedelta(hours=5)  # Convert UTC to EST
    
    if language == 'fr':
        title = 'CONCENTRATION DE FUMÉE DE FEU DE FORÊT EN AMÉRIQUE DU NORD'
        month_names_fr = {
            'January': 'janvier', 'February': 'février', 'March': 'mars', 'April': 'avril',
            'May': 'mai', 'June': 'juin', 'July': 'juillet', 'August': 'août',
            'September': 'septembre', 'October': 'octobre', 'November': 'novembre', 'December': 'décembre'
        }
        month_en = est_time.strftime("%B")
        month_fr = month_names_fr.get(month_en, month_en)
        subtitle = f'Concentrations de PM2.5 en Surface - {est_time.day} {month_fr} {est_time.year} à {est_time.strftime("%H:%M")} EST'
        data_note = "DONNÉES: FIRESMOKE.CA / ENVIRONNEMENT ET CHANGEMENT CLIMATIQUE CANADA"
    else:
        title = 'WILDFIRE SMOKE CONCENTRATION ACROSS NORTH AMERICA'
        subtitle = f'PM2.5 Surface Concentrations - {est_time.strftime("%B %d, %Y at %I:%M %p EST")}'
        data_note = "DATA: FIRESMOKE.CA / ENVIRONMENT AND CLIMATE CHANGE CANADA"
    
    # Add title and subtitle aligned with map left edge (at x=0.02, same as map)
    fig.text(0.02, 0.96, title, fontsize=18, weight='bold', color='#333333',
             transform=fig.transFigure, family='monospace')
    
    fig.text(0.02, 0.93, subtitle, fontsize=14, color='#666666',
             transform=fig.transFigure, family='monospace')
    
    # Add data source note aligned with map left edge
    fig.text(0.02, 0.02, f"ANALYSIS: DEEP SKY RESEARCH\n{data_note}",
             fontsize=10, color='#888888',
             transform=fig.transFigure,
             family='monospace',
             verticalalignment='bottom')
    
    # Add Deep Sky logo in bottom right corner
    try:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.image as mpimg
        
        # Use the favicon path from utils.py
        logo_path = '/Users/max/Deep_Sky/design/Favicon/favicon_for_charts.png'
        
        if os.path.exists(logo_path):
            # Load and add the logo
            logo_img = mpimg.imread(logo_path)
            imagebox = OffsetImage(logo_img, zoom=0.03)
            
            # Position at bottom right (x=0.98, y=0.02)
            ab = AnnotationBbox(imagebox, (0.98, 0.02),
                              xycoords='figure fraction',
                              box_alignment=(1.0, 0.0),  # Align to bottom-right of the box
                              frameon=False)
            fig.add_artist(ab)
            print("Added Deep Sky favicon logo")
        else:
            # Fallback to text logo if image not found
            fig.text(0.98, 0.02, "DEEP SKY",
                    fontsize=12, weight='bold', color='#333333',
                    transform=fig.transFigure,
                    family='monospace',
                    horizontalalignment='right',
                    verticalalignment='bottom')
            print(f"Logo file not found at {logo_path}, using text fallback")
            
    except Exception as e:
        # Final fallback to text logo
        fig.text(0.98, 0.02, "DEEP SKY",
                fontsize=12, weight='bold', color='#333333',
                transform=fig.transFigure,
                family='monospace',
                horizontalalignment='right',
                verticalalignment='bottom')
        print(f"Could not load logo: {e}, using text fallback")
    
    # Add intensity badge on the map itself
    max_pm25 = float(pm25_data.max().values)
    if language == 'fr':
        if max_pm25 > 250:
            intensity = "DANGEREUX"
            color = '#8b0000'
        elif max_pm25 > 150:
            intensity = "TRÈS MALSAIN"
            color = '#a0522d'
        elif max_pm25 > 55:
            intensity = "MALSAIN"
            color = '#c67c3b'
        else:
            intensity = "MODÉRÉ"
            color = '#d4a574'
        intensity_label = f'POINTE: {intensity}\n({max_pm25:.0f} μg/m³)'
    else:
        if max_pm25 > 250:
            intensity = "HAZARDOUS"
            color = '#8b0000'
        elif max_pm25 > 150:
            intensity = "VERY UNHEALTHY"
            color = '#a0522d'
        elif max_pm25 > 55:
            intensity = "UNHEALTHY"
            color = '#c67c3b'
        else:
            intensity = "MODERATE"
            color = '#d4a574'
        intensity_label = f'PEAK: {intensity}\n({max_pm25:.0f} μg/m³)'
    
    # Add intensity badge positioned relative to map
    ax.text(0.02, 0.98, intensity_label, 
            transform=ax.transAxes, fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='white'),
            color='white', verticalalignment='top', family='monospace')
    
    # Save with higher DPI for better quality - no tight_layout needed since we control positioning
    save_plot(fig, output_file)
    print(f"Enhanced smoke map saved as '{output_file}'")
    
    return fig

def analyze_smoke_statistics(pm25_data, timestamp):
    """Calculate and print smoke concentration statistics"""
    print(f"\n--- Smoke Analysis for {timestamp.strftime('%Y-%m-%d %H:%M UTC')} ---")
    
    # Remove NaN values for statistics
    valid_data = pm25_data.values[~np.isnan(pm25_data.values)]
    
    if len(valid_data) == 0:
        print("No valid smoke data available")
        return
    
    # Calculate statistics
    stats = {
        'mean': np.mean(valid_data),
        'median': np.median(valid_data),
        'max': np.max(valid_data),
        'min': np.min(valid_data),
        'std': np.std(valid_data)
    }
    
    print(f"Mean PM2.5: {stats['mean']:.2f} μg/m³")
    print(f"Median PM2.5: {stats['median']:.2f} μg/m³")
    print(f"Maximum PM2.5: {stats['max']:.2f} μg/m³")
    print(f"Minimum PM2.5: {stats['min']:.2f} μg/m³")
    print(f"Standard Deviation: {stats['std']:.2f} μg/m³")
    
    # Calculate percentage of area in each AQI category
    boundaries = [0, 12, 35, 55, 150, 250, 500]
    categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                  'Unhealthy', 'Very Unhealthy', 'Hazardous']
    
    print(f"\nAir Quality Distribution:")
    total_points = len(valid_data)
    
    for i, (category, lower, upper) in enumerate(zip(categories, boundaries[:-1], boundaries[1:])):
        if i == len(categories) - 1:  # Last category
            count = np.sum(valid_data >= lower)
        else:
            count = np.sum((valid_data >= lower) & (valid_data < upper))
        
        percentage = (count / total_points) * 100
        print(f"  {category}: {percentage:.1f}% ({count:,} grid cells)")

def create_time_series_plot(ds, output_file='smoke_time_series.png', language='en'):
    """Create a time series plot of average smoke concentrations"""
    print(f"Creating time series plot in {language.upper()}...")
    
    # Process all time steps
    datetime_list = decode_time_flags(ds['TFLAG'])
    pm25 = ds['PM25']
    
    # Calculate spatial average for each time step
    avg_concentrations = []
    for i in range(len(datetime_list)):
        timestep_data = pm25.isel(TSTEP=i, LAY=0)
        valid_data = timestep_data.values[~np.isnan(timestep_data.values)]
        if len(valid_data) > 0:
            avg_concentrations.append(np.mean(valid_data))
        else:
            avg_concentrations.append(np.nan)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot time series
    ax.plot(datetime_list, avg_concentrations, 
           color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
    
    # Set labels based on language
    if language == 'fr':
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Concentration Moyenne de PM2.5 (μg/m³)', fontsize=12)
        title = 'CONCENTRATIONS MOYENNES DE FUMÉE EN AMÉRIQUE DU NORD'
        subtitle = f'Moyenne Spatiale PM2.5 - {datetime_list[0].strftime("%d %B")} au {datetime_list[-1].strftime("%d %B %Y")}'
        data_note = "DONNÉES: FIRESMOKE.CA / ENVIRONNEMENT ET CHANGEMENT CLIMATIQUE CANADA"
    else:
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average PM2.5 Concentration (μg/m³)', fontsize=12)
        title = 'NORTH AMERICA AVERAGE SMOKE CONCENTRATIONS'
        subtitle = f'Spatial Average PM2.5 - {datetime_list[0].strftime("%B %d")} to {datetime_list[-1].strftime("%B %d, %Y")}'
        data_note = "DATA: FIRESMOKE.CA / ENVIRONMENT AND CLIMATE CHANGE CANADA"
    
    ax.grid(True, alpha=0.3)
    
    # Get font properties
    font_props = setup_space_mono_font()
    
    # Format title
    format_plot_title(ax, title, subtitle, font_props)
    
    # Add branding
    add_deep_sky_branding(ax, font_props, data_note=data_note)
    
    # Save the plot
    save_plot(fig, output_file)
    print(f"Time series plot saved as '{output_file}'")
    
    return fig

def main():
    """Main function to process and visualize smoke data"""
    # File path
    smoke_nc = 'smoke/dispersion.nc'
    
    try:
        # Load the data
        ds = load_smoke_data(smoke_nc)
        
        # Process the smoke data
        latest_pm25, timestamp = process_smoke_data(ds)
        
        # Estimate coordinates
        lons, lats = estimate_coordinates(ds)
        
        # Create output directory
        import os
        os.makedirs('figures/smoke', exist_ok=True)
        
        # Generate English smoke map
        plot_smoke_map(latest_pm25, timestamp, lons, lats, 
                      'figures/smoke/smoke_map_north_america.png', language='en')
        
        # Generate French smoke map
        plot_smoke_map(latest_pm25, timestamp, lons, lats, 
                      'figures/smoke/smoke_map_north_america_fr.png', language='fr')
        
        # # Analyze statistics
        # analyze_smoke_statistics(latest_pm25, timestamp)
        
        # # Create English time series plot
        # create_time_series_plot(ds, 'figures/smoke/smoke_time_series.png', language='en')
        
        # # Create French time series plot
        # create_time_series_plot(ds, 'figures/smoke/smoke_time_series_fr.png', language='fr')
        
    except Exception as e:
        print(f"Error processing smoke data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()