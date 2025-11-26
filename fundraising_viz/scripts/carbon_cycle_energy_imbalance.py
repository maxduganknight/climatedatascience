"""
Earth's Energy Imbalance Analysis

This script reads NASA CERES EBAF data and creates a visualization showing
the Earth's energy imbalance by plotting Absorbed Solar Radiation (ASR) and
Outgoing Longwave Radiation (OLR) over time.

Data source: NASA CERES EBAF Edition 4.2 (March 2000 - July 2024)
"""

import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
import sys
import datetime
from scipy.ndimage import uniform_filter1d

sys.path.append('../reports')
from utils import setup_enhanced_plot, format_plot_title, add_deep_sky_branding, save_plot


def load_ceres_data(file_path, cache_path='data/carbon_cycle/ceres_processed_cache.csv'):
    """
    Load CERES EBAF data from NetCDF file or cached CSV.

    If a cached CSV exists, loads from cache (fast).
    Otherwise, processes the NetCDF file and saves to cache.

    Args:
        file_path: Path to the original NetCDF file
        cache_path: Path to the cached CSV file (default: data/carbon_cycle/ceres_processed_cache.csv)

    Returns:
        DataFrame with time, ASR (Absorbed Solar Radiation), and OLR (Outgoing Longwave Radiation)
    """
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f'Loading CERES data from cache: {cache_path}')
        df = pd.read_csv(cache_path, parse_dates=['date'])

        print(f'  ✓ Loaded {len(df)} monthly data points from cache')
        print(f'  Date range: {df["date"].min()} to {df["date"].max()}')
        print(f'  ASR range: {df["ASR"].min():.2f} to {df["ASR"].max():.2f} W/m²')
        print(f'  OLR range: {df["OLR"].min():.2f} to {df["OLR"].max():.2f} W/m²')

        return df

    # Cache doesn't exist - process NetCDF file
    print(f'Cache not found. Processing NetCDF file: {file_path}')
    print('(This is a one-time operation - subsequent runs will use the cache)')

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"NetCDF file not found: {file_path}\n"
            f"Please download from: https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF_Edition4.2\n"
            f"Or if you already processed the data, check for cache at: {cache_path}"
        )

    print('Loading CERES EBAF data from NetCDF...')
    ds = nc.Dataset(file_path)

    # Get time variable and convert to dates
    time_var = ds.variables['time']
    time_values = time_var[:]
    time_units = time_var.units  # Should be something like "days since 2000-03-01"

    # Convert time to datetime
    dates = nc.num2date(time_values, units=time_units, calendar='standard')
    # Convert cftime objects to pandas timestamps (only year, month, day to avoid time precision issues)
    dates = [pd.Timestamp(d.year, d.month, d.day) for d in dates]

    # Get global mean values
    # ASR = Incoming Solar - Reflected Solar = solar_mon - toa_sw_all_mon
    # For global means, we use the 'g' prefix variables
    solar_in = ds.variables['gsolar_mon'][:]  # Incoming solar
    toa_sw = ds.variables['gtoa_sw_all_mon'][:]  # Reflected shortwave at TOA
    toa_lw = ds.variables['gtoa_lw_all_mon'][:]  # Outgoing longwave at TOA

    # ASR = Incoming solar - Reflected solar
    asr = solar_in - toa_sw

    # OLR = Outgoing longwave
    olr = toa_lw

    ds.close()

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'ASR': asr,
        'OLR': olr
    })

    # Remove any rows with invalid values (NaN, inf, or unreasonable values)
    # Valid range for ASR and OLR should be roughly 200-300 W/m²
    df = df[(df['ASR'] > 200) & (df['ASR'] < 300) &
            (df['OLR'] > 200) & (df['OLR'] < 300)].copy()

    # Explicitly trim to end at July 2024 to avoid any edge effects
    df = df[df['date'] <= pd.Timestamp('2024-07-31')].copy()

    # Reset index after filtering
    df = df.reset_index(drop=True)

    print(f'  Loaded {len(df)} monthly data points')
    print(f'  Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'  ASR range: {df["ASR"].min():.2f} to {df["ASR"].max():.2f} W/m²')
    print(f'  OLR range: {df["OLR"].min():.2f} to {df["OLR"].max():.2f} W/m²')

    # Save to cache
    print(f'\nSaving processed data to cache: {cache_path}')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f'  ✓ Cache saved. You can now delete the original NetCDF file to save space.')
    print(f'  NetCDF file location: {file_path}')
    print(f'  Cache file location: {cache_path}')

    return df


def calculate_12_month_mean(series):
    """
    Calculate 12-month running mean.
    Only calculates where we have a full 12 months of data.
    Returns NaN for positions where we don't have enough data.
    """
    result = np.full(len(series), np.nan)

    # Only calculate 12-month mean where we have sufficient data
    # For centered window: need 6 months before and after
    # For trailing window: need 11 months before
    # We'll use a trailing 12-month window
    for i in range(11, len(series)):
        result[i] = np.mean(series[i-11:i+1])

    return result


def calculate_48_month_mean(series):
    """
    Calculate 48-month running mean.
    Only calculates where we have a full 48 months of data.
    Returns NaN for positions where we don't have enough data.
    """
    result = np.full(len(series), np.nan)

    # Only calculate 48-month mean where we have sufficient data
    # We'll use a trailing 48-month window
    for i in range(47, len(series)):
        result[i] = np.mean(series[i-47:i+1])

    return result


def calculate_trend(df, column):
    """
    Calculate polynomial trend line for a time series.
    Uses 2nd degree polynomial for a slight curve.
    """
    # Convert dates to numeric values (days since first date)
    x = (df['date'] - df['date'].min()).dt.days.values
    y = df[column].values

    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Fit polynomial trend (degree 2 for slight curve)
    coeffs = np.polyfit(x_clean, y_clean, 2)
    trend = np.polyval(coeffs, x)

    return trend


def create_energy_imbalance_plot(df):
    """
    Create the Earth's Energy Imbalance plot.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 10))

    font_prop = font_props.get('regular') if font_props else None
    font_bold = font_props.get('bold') if font_props else None

    # Calculate 12-month running means
    df['ASR_12m'] = calculate_12_month_mean(df['ASR'].values)
    df['OLR_12m'] = calculate_12_month_mean(df['OLR'].values)

    # Calculate trends
    df['ASR_trend'] = calculate_trend(df, 'ASR_12m')
    df['OLR_trend'] = calculate_trend(df, 'OLR_12m')

    # Ensure we only plot valid data points (no NaN)
    valid_mask = ~(df['ASR_12m'].isna() | df['OLR_12m'].isna())
    df_plot = df[valid_mask].copy()

    # Plot the data
    # ASR - black line
    ax.plot(df_plot['date'], df_plot['ASR_12m'], color='black', linewidth=2,
            label='Absorbed Solar Radiation (ASR)', zorder=3, solid_capstyle='round')

    # OLR - red line
    ax.plot(df_plot['date'], df_plot['OLR_12m'], color='#DC143C', linewidth=2,
            label='Outgoing Longwave Radiation (OLR)', zorder=3, solid_capstyle='round')

    # Trend lines - dashed (only plot where we have valid data)
    ax.plot(df_plot['date'], df_plot['ASR_trend'], color='black', linewidth=2,
            linestyle='--', alpha=0.7, zorder=2)

    ax.plot(df_plot['date'], df_plot['OLR_trend'], color='#DC143C', linewidth=2,
            linestyle='--', alpha=0.7, zorder=2)

    # Highlight El Niño periods (beige shading)
    # Major El Niño events: 2002-2003, 2009-2010, 2015-2016, 2023-2024
    # Ensure we don't shade beyond our data range
    max_date = df_plot['date'].max()
    # el_nino_periods = [
    #     (pd.Timestamp('2002-06-01'), pd.Timestamp('2003-06-01')),
    #     (pd.Timestamp('2009-06-01'), pd.Timestamp('2010-06-01')),
    #     (pd.Timestamp('2015-01-01'), pd.Timestamp('2016-06-01')),
    #     (pd.Timestamp('2023-06-01'), min(pd.Timestamp('2024-06-01'), max_date))
    # ]

    # for start, end in el_nino_periods:
    #     ax.axvspan(start, end, color='#F5DEB3', alpha=0.3, zorder=1)

    # Add El Niño label annotation
    # ax.text(pd.Timestamp('2018-01-01'), 242.3, '12-month\nEl Niño periods\n~0.42 W/m²',
    #         fontsize=10, ha='center', va='center', fontproperties=font_prop,
    #         color='#8B7355')

    # Calculate energy imbalance values at start and end
    # Start value (around 2015)
    start_year = pd.Timestamp('2015-01-01')
    start_idx = (df_plot['date'] - start_year).abs().idxmin()
    start_imbalance = df_plot['ASR_trend'].iloc[start_idx] - df_plot['OLR_trend'].iloc[start_idx]
    start_asr = df_plot['ASR_trend'].iloc[start_idx]
    start_olr = df_plot['OLR_trend'].iloc[start_idx]

    # End value (at 2024 with actual year-to-date average)
    # Use the last available data point (July 2024)
    end_idx = -2
    end_year = df_plot['date'].iloc[end_idx]
    end_imbalance = df_plot['ASR_12m'].iloc[end_idx] - df_plot['OLR_12m'].iloc[end_idx]
    end_asr = df_plot['ASR_12m'].iloc[end_idx]
    end_olr = df_plot['OLR_12m'].iloc[end_idx]

    # Add imbalance annotations with vertical lines and notches
    # Start annotation (2015)
    offset = 0.05  # Offset for notches

    # Vertical line from OLR to ASR
    ax.plot([df_plot['date'].iloc[start_idx], df_plot['date'].iloc[start_idx]],
            [start_olr - offset, start_asr + offset],
            color='#E74C3C', linewidth=3, alpha=0.9, zorder=4)

    # Top notch (at ASR level)
    notch_days = pd.Timedelta(days=60)
    ax.plot([df_plot['date'].iloc[start_idx] - notch_days, df_plot['date'].iloc[start_idx] + notch_days],
            [start_asr + offset, start_asr + offset],
            color='#E74C3C', linewidth=3, alpha=0.9, zorder=4)

    # Bottom notch (at OLR level)
    ax.plot([df_plot['date'].iloc[start_idx] - notch_days, df_plot['date'].iloc[start_idx] + notch_days],
            [start_olr - offset, start_olr - offset],
            color='#E74C3C', linewidth=3, alpha=0.9, zorder=4)

    # Text annotation for start
    ax.text(df_plot['date'].iloc[start_idx], start_asr + 0.4, f'+{start_imbalance:.2f}\nW/m²',
            fontsize=11, fontproperties=font_bold, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=1.5))

    # End annotation (2024)
    # Vertical line from OLR to ASR
    ax.plot([end_year, end_year],
            [end_olr - offset, end_asr + offset],
            color='#E74C3C', linewidth=3, alpha=0.9, zorder=4)

    # Top notch (at ASR level)
    ax.plot([end_year - notch_days, end_year + notch_days],
            [end_asr + offset, end_asr + offset],
            color='#E74C3C', linewidth=3, alpha=0.9, zorder=4)

    # Bottom notch (at OLR level)
    ax.plot([end_year - notch_days, end_year + notch_days],
            [end_olr - offset, end_olr - offset],
            color='#E74C3C', linewidth=3, alpha=0.9, zorder=4)

    # Text annotation for end
    ax.text(end_year, end_asr + 0.4, f'+{end_imbalance:.2f}\nW/m²',
            fontsize=11, fontproperties=font_bold, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=1.5))

    # Formatting
    ax.set_ylabel('Watts per square meter (W/m²)', fontsize=13,
                  fontproperties=font_prop, labelpad=10)
    ax.set_xlabel('Year', fontsize=13, fontproperties=font_prop, labelpad=10)

    # Set axis limits - use actual data range
    # Set x-axis to exactly the data range without extension
    ax.set_xlim(df_plot['date'].min(), df_plot['date'].max())
    ax.set_ylim(239.5, 243.0)

    # Add legend
    ax.legend(loc='lower left', fontsize=11, frameon=True,
             facecolor='white', edgecolor='#DDDDDD', prop=font_prop)

    # Grid
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    return fig


def create_energy_imbalance_48m_plot(df):
    """
    Create the Earth's Energy Imbalance plot with 48-month running mean.
    """
    fig, ax, font_props = setup_enhanced_plot(figsize=(16, 10))

    font_prop = font_props.get('regular') if font_props else None
    font_bold = font_props.get('bold') if font_props else None

    # Calculate 48-month running means
    df['ASR_48m'] = calculate_48_month_mean(df['ASR'].values)
    df['OLR_48m'] = calculate_48_month_mean(df['OLR'].values)

    # Calculate trends
    df['ASR_trend'] = calculate_trend(df, 'ASR_48m')
    df['OLR_trend'] = calculate_trend(df, 'OLR_48m')

    # Ensure we only plot valid data points (no NaN)
    valid_mask = ~(df['ASR_48m'].isna() | df['OLR_48m'].isna())
    df_plot = df[valid_mask].copy()

    # Plot the data
    # ASR - black line
    ax.plot(df_plot['date'], df_plot['ASR_48m'], color='black', linewidth=2,
            label='Absorbed Solar Radiation (ASR)', zorder=3, solid_capstyle='round')

    # OLR - red line
    ax.plot(df_plot['date'], df_plot['OLR_48m'], color='#DC143C', linewidth=2,
            label='Outgoing Longwave Radiation (OLR)', zorder=3, solid_capstyle='round')

    # Trend lines - dashed (only plot where we have valid data)
    # ax.plot(df_plot['date'], df_plot['ASR_trend'], color='black', linewidth=2,
    #         linestyle='--', alpha=0.7, zorder=2)

    # ax.plot(df_plot['date'], df_plot['OLR_trend'], color='#DC143C', linewidth=2,
    #         linestyle='--', alpha=0.7, zorder=2)

    # Calculate energy imbalance values at start and end of 48-month data
    # Start value (first point with 48-month data)
    start_idx = 0
    start_imbalance = df_plot['ASR_48m'].iloc[start_idx] - df_plot['OLR_48m'].iloc[start_idx]
    start_asr = df_plot['ASR_48m'].iloc[start_idx]
    start_olr = df_plot['OLR_48m'].iloc[start_idx]
    start_date = df_plot['date'].iloc[start_idx]
    # Calculate the 48-month range for the start (47 months before this point)
    start_range_begin = df_plot['date'].iloc[0] - pd.DateOffset(months=47)
    start_range_end = df_plot['date'].iloc[0]
    start_label = f"{start_range_begin.year}-{start_range_end.year}"

    # End value (last point with 48-month data)
    end_idx = -1
    end_imbalance = df_plot['ASR_48m'].iloc[end_idx] - df_plot['OLR_48m'].iloc[end_idx]
    end_asr = df_plot['ASR_48m'].iloc[end_idx]
    end_olr = df_plot['OLR_48m'].iloc[end_idx]
    end_date = df_plot['date'].iloc[end_idx]
    # Calculate the 48-month range for the end (47 months before this point)
    end_range_begin = df_plot['date'].iloc[end_idx] - pd.DateOffset(months=47)
    end_range_end = df_plot['date'].iloc[end_idx]
    end_label = f"{end_range_begin.year}-{end_range_end.year}"

    # Add imbalance annotations with vertical lines and notches
    offset = 0.05  # Offset for notches
    notch_days = pd.Timedelta(days=45)
    start_gap = df_plot['date'].iloc[3]
    end_gap = df_plot['date'].iloc[-4]

    # Start annotation
    # Vertical line from OLR to ASR
    ax.plot([start_gap, start_gap],
            [start_olr + offset, start_asr - offset],
            color='#E74C3C', linewidth=2, alpha=0.9, zorder=4)

    # Top notch (at ASR level)
    ax.plot([start_gap - notch_days, start_gap + notch_days],
            [start_asr - offset, start_asr - offset],
            color='#E74C3C', linewidth=2, alpha=0.9, zorder=4)

    # Bottom notch (at OLR level)
    ax.plot([start_gap - notch_days, start_gap + notch_days],
            [start_olr + offset, start_olr + offset],
            color='#E74C3C', linewidth=2, alpha=0.9, zorder=4)

    # Text annotation for start
    ax.text(start_gap + notch_days * 10, (start_asr + start_olr)/2, f"ENERGY IMBALANCE:\n+{start_imbalance:.2f} W/m²",
            fontsize=10, fontproperties=font_bold, ha='center', va='bottom',
            # bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
            #          edgecolor='black', linewidth=1.5)
                     )

    # End annotation
    # Vertical line from OLR to ASR
    ax.plot([end_gap, end_gap],
            [end_olr + offset, end_asr - offset],
            color='#E74C3C', linewidth=2, alpha=0.9, zorder=4)

    # Top notch (at ASR level)
    ax.plot([end_gap - notch_days, end_gap + notch_days],
            [end_asr - offset, end_asr - offset],
            color='#E74C3C', linewidth=2, alpha=0.9, zorder=4)

    # Bottom notch (at OLR level)
    ax.plot([end_gap - notch_days, end_gap + notch_days],
            [end_olr + offset, end_olr + offset],
            color='#E74C3C', linewidth=2, alpha=0.9, zorder=4)

    # Text annotation for end
    ax.text(end_gap - notch_days * 10, (end_asr + start_asr)/2, f"ENERGY IMBALANCE:\n +{end_imbalance:.2f} W/m²",
            fontsize=10, fontproperties=font_bold, ha='center', va='bottom')
            # bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
            #          edgecolor='black', linewidth=1.5))
    
    # annotate lines
    ax.text(pd.to_datetime("2017,01,01"), 240.3, "OUTGOING LONGWAVE RADIATION",
            fontsize=12, ha='left', va='bottom',
            color = '#DC143C')
    
    ax.text(pd.to_datetime("2013,01,01"), 241.35, "ABSORBED SOLAR RADIATION",
        fontsize=12, ha='left', va='bottom',
        color = 'black')

    # Formatting
    # ax.set_ylabel()
    # ax.set_xlabel()

    # Set axis limits - use actual data range
    ax.set_xlim(df_plot['date'].min(), df_plot['date'].max())
    ax.set_ylim(240, 242.3)

    # Add legend
    # ax.legend(loc='lower left', fontsize=11, frameon=True,
    #          facecolor='white', edgecolor='#DDDDDD', prop=font_prop)

    # Grid
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    return fig


def main():
    """
    Main execution function.
    """
    print('='*60)
    print('Earth\'s Energy Imbalance Analysis')
    print('='*60)

    # Load data
    # if file doesn't exist you have to redownload it from here: https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF_Edition4.2
    data_file = 'data/carbon_cycle/CERES_EBAF_Edition4.2_200003-202407.nc'
    df = load_ceres_data(data_file)

    # Create visualization
    print('\nCreating energy imbalance visualization...')
    fig = create_energy_imbalance_plot(df)

    # Add title and branding
    format_plot_title(plt.gca(),
                     "",
                     "Absorbed Solar Radiation (ASR) vs Outgoing Longwave Radiation (OLR)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: NASA CERES EBAF-TOA All-sky Ed4.2 Solar - SW & OLW, 2000/03-2024/07, 12-month mean",
                         analysis_date=datetime.datetime.now())

    # Save plot
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    save_plot(fig, 'figures/earth_energy_imbalance.png')

    print('  ✓ Saved to figures/earth_energy_imbalance.png')

    # Create 48-month version
    print('\nCreating 48-month energy imbalance visualization...')
    fig_48m = create_energy_imbalance_48m_plot(df.copy())

    # Add title and branding
    format_plot_title(plt.gca(),
                     "",
                     "WATTS PER SQUARE METER (W/m²)",
                     None)

    add_deep_sky_branding(plt.gca(), None,
                         "DATA: NASA CERES EBAF-TOA All-sky Ed4.2 Solar - SW & OLW, 2000/03-2024/07, 48-month mean. Amended from Leon Simons.",
                         analysis_date=datetime.datetime.now())

    save_plot(fig_48m, 'figures/earth_energy_imbalance_48m.png')

    print('  ✓ Saved to figures/earth_energy_imbalance_48m.png')
    print('\n' + '='*60)
    print('✓ Complete!')
    print('='*60)


if __name__ == '__main__':
    main()
