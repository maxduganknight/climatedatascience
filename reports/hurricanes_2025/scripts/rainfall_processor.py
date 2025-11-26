import pandas as pd
import numpy as np
import xarray as xr

# viz
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import plotly.graph_objects as go
import plotly.express as px
import folium
import seaborn as sns

import glob
import sys
import os

# Add reports directory to Python path to import shared utilities
reports_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, reports_dir)

# Import shared utilities
from utils import (
    setup_enhanced_plot, format_plot_title, add_deep_sky_branding,
    save_plot, COLORS
)

from scipy.interpolate import griddata
from numpy.polynomial.polynomial import Polynomial

import geopandas as gpd
import rioxarray
from shapely.geometry import mapping

# functions for state masking

def get_shape_us(shapefile, state_abbreviation):
    shapefile = shapefile.to_crs("EPSG:4326")
    state_shape = shapefile[shapefile.STUSPS == state_abbreviation].geometry
    return state_shape

def mask_xr_dataset(xr_data, shape):
    xr_data = xr_data.assign_coords(longitude=(((xr_data.longitude + 180) % 360) - 180)).sortby('longitude')
    xr_data.rio.write_crs("EPSG:4326", inplace=True)
    xr_data_masked = xr_data.rio.clip(shape.geometry.apply(mapping), shape.crs)
    return xr_data_masked

def fit_polynomial(df):
    # add trend line
    p = Polynomial.fit(df.index, df['tp'], 1)
    print('Slope: ', p.convert().coef[1])
    x_values = np.linspace(df.index.min(), df.index.max(), len(df.index))
    y_values = p(x_values)
    df['trend'] = y_values
    df = df[['year', 'tp', 'trend']]
    return df

def calculate_monthly_maxes(ERA5_land):
    ERA5_coarsened = ERA5_land.coarsen(latitude=5, longitude=5, boundary='trim').sum()
    ERA5_agg = ERA5_coarsened.resample(valid_time='1D').sum()
    ERA5_agg_monthly = ERA5_agg['tp'].resample(valid_time='ME').max()
    ERA5_agg_monthly_max = ERA5_agg_monthly.max(['latitude', 'longitude'])
    ERA5_agg_monthly_max_df = ERA5_agg_monthly_max.to_dataframe().reset_index()
    filtered_df = ERA5_agg_monthly_max_df.loc[ERA5_agg_monthly_max_df['valid_time'].dt.month.between(6, 10)]
    filtered_df['year'] = filtered_df['valid_time'].dt.year
    filtered_df.to_csv('rainfall_data/ERA5_1d_monthly_max.csv', index=False)

def calculate_annual_999_frequency(ERA5_agg):
    ERA5_agg = ERA5_agg.chunk({'valid_time': -1})
    extreme_precipitation_treshold = ERA5_agg.sel(valid_time=slice(None, '2000')).tp.quantile(.999, dim=['valid_time'])
    count_above_threshold = xr.where(ERA5_agg['tp'] > extreme_precipitation_treshold, 1, 0)
    ERA5_agg_annual = count_above_threshold.resample(valid_time='YE').sum()
    ERA5_agg_annual_total = ERA5_agg_annual.sum(['latitude', 'longitude'])
    ERA5_extreme_frequency_df = ERA5_agg_annual_total.to_dataframe().reset_index()
    ERA5_extreme_frequency_df['year'] = ERA5_extreme_frequency_df['valid_time'].dt.year
    plot_df = fit_polynomial(ERA5_extreme_frequency_df)
    plot_df.to_csv('rainfall_data/ERA5_999_counts.csv', index=False)
    return plot_df

def create_extreme_rainfall_chart(df, save_path):
    """
    Create a bar chart visualization for extreme hurricane rainfall days.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with columns: year, tp (extreme rainfall count), trend
    save_path : str
        Path to save the chart
    """
    print(f"Creating extreme rainfall chart...")

    # Set up the plot with common styling
    fig, ax, font_props = setup_enhanced_plot(figsize=(15, 10))

    # Create the bar chart with purple color matching the screenshot
    purple_color = '#8B5A96'  # Purple color similar to the screenshot
    bars = ax.bar(df['year'], df['tp'],
                  color=purple_color, width=0.8, alpha=0.8)

    # Format the plot title and subtitle
    title = "Extreme Rainfall is Becoming More Common"
    subtitle = "Extreme Rainfall Occurences During Hurricane Season in Southeastern US"
    format_plot_title(ax, title, subtitle, font_props)

    # Format axes
    font_prop = font_props.get('regular') if font_props else None

    # X-axis: Show every 5th year for readability, matching the screenshot
    years = df['year'].values
    x_ticks = years[::5]  # Every 5th year
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontproperties=font_prop, fontsize=14)

    # Y-axis formatting
    ax.set_ylim(0, df['tp'].max() * 1.1)  # Add some padding at top
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter('{x:.0f}')

    # Remove y-axis label to match the screenshot style
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Add branding and data note
    data_note = "1-day precipitation totals above the 99.9th percentile.\nDATA: ERA5, C3S/ECMWF"
    add_deep_sky_branding(ax, font_props, data_note=data_note)

    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot
    save_plot(fig, save_path)

    print(f"Extreme rainfall chart saved to: {save_path}")

    return fig

if __name__ == '__main__':

    ERA5 = xr.open_mfdataset('rainfall_data/rainfall_raw/ERA5_????.nc', combine='by_coords')
    lat_era5 = ERA5.latitude.values
    lon_era5 = ERA5.longitude.values

    LSMask = xr.open_dataset('rainfall_data/ERA5_land_sea_mask_new.nc')
    LSMask['longitude'] = (((LSMask['longitude'] + 180) % 360) - 180)
    LSMask_interp = LSMask.interp(latitude=lat_era5, longitude=lon_era5)

    # uncomment below lines to filter for a specific state

    # us_shapefile = gpd.read_file('../../data/shapefiles/us/states/tl_2024_us_state/tl_2024_us_state.shp')
    # shape = get_shape_us(us_shapefile, 'FL')
    # filtered_era5 = mask_xr_dataset(ERA5, shape)

    ERA5_land = (
        ERA5.where(LSMask_interp['lsm'].sel(time = '1981').squeeze('time') > 0.5)
    )

    # area_weights = np.cos(np.deg2rad(filtered_era5.latitude))
    # area_weights = area_weights.expand_dims({'longitude': filtered_era5.longitude})

    #filtered_era5['tp'] = filtered_era5['tp'] * area_weights

    ERA5_coarsened = ERA5_land.coarsen(latitude=3, longitude=3, boundary='trim').sum()
    ERA5_agg = ERA5_coarsened.resample(valid_time='1D').sum()

    # Calculate extreme rainfall frequency and get the DataFrame
    extreme_rainfall_df = calculate_annual_999_frequency(ERA5_agg)

    # Create the visualization
    create_extreme_rainfall_chart(extreme_rainfall_df, 'figures/rainfall/extreme_rainfall_southeastern_us.png')





