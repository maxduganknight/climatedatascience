import pandas as pd
import numpy as np
import xarray as xr
import cdsapi 
import glob
import sys
import os
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import argparse
import warnings

warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed")

def smooth_values(data):
    ''' 
    As the raw values are subject to noise due to the limited sampling, 
    they are smoothed by Fourier filtering with a wavenumber eight truncation.
    '''
    data = np.asarray(data)
    fft_coeffs = np.fft.fft(data)
    fft_coeffs[8:] = 0
    smoothed_data = np.fft.ifft(fft_coeffs).real
    return smoothed_data 

def calculate_baseline(data, start_year, end_year, weekly=False):
    '''
    Calculate the baseline as the mean temperature over the given time period.
    Smooth the daily values using the Fourier filter.
    '''
    unit = 'week' if weekly else 'month_day'
    data = data.sortby('valid_time')
    period = data.sel(valid_time=slice(start_year, end_year))
    means = period.groupby(unit).mean()
    smoothed_means = xr.apply_ufunc(
        smooth_values,
        means,
        input_core_dims=[[unit]],
        output_core_dims=[[unit]],
        vectorize=True
    )
    if weekly:
        smoothed_means = smoothed_means.sel(week=data[unit])
    else:
        smoothed_means = smoothed_means.sel(month_day=data[unit])
    return smoothed_means

def preprocess(data, variable, weekly=False):
    print(f'Preprocessing {variable} data') 
    time_step = 'week' if weekly else 'month_day'
    temp = data[variable]
    temp_c = temp - 273.15
    temp_c = temp_c.assign_attrs(temp.attrs)
    temp_c.attrs['units'] = '° C'
    if weekly:
        temp_c['week'] = temp_c['valid_time'].dt.isocalendar().week
        baseline = calculate_baseline(temp_c, '1979-01-01', '2000-12-31', weekly=True)
    else:
        temp_c['month_day'] = temp_c['valid_time'].dt.strftime('%m-%d')
        baseline = calculate_baseline(temp_c, '1979-01-01', '2000-12-31', weekly=False)

    anom = temp_c - baseline
    anom_normalized = anom.groupby(time_step) - anom.groupby(time_step).mean()
    temp_df = clean_to_df(anom_normalized, temp_c, baseline, variable, weekly)
    print(f'Finished preprocessing {variable} data')
    return temp_df

# def preprocess_pre_industrial_baseline(data, variable, baseline_df):
#     print(f'Preprocessing {variable} data') 
#     temp = data[variable]
#     temp_c = temp - 273.15
#     temp_c = temp_c.assign_attrs(temp.attrs)
#     temp_c.attrs['units'] = '° C'
#     temp_c.name = 't2m'
#     temp_c_df = temp_c.to_dataframe().reset_index()
#     temp_c_df['month'] = temp_c_df['valid_time'].dt.month
#     print("TEMP C DF:")
#     print(temp_c_df)
#     merged_df = pd.merge(temp_c_df, baseline_df, on='month', how='left')
#     merged_df['anom'] = merged_df['t2m'] - merged_df['baseline']
#     print("MERGED DF:")
#     print(merged_df)
#     #anom_normalized = merged_df.groupby('month') - merged_df.groupby('month').mean()
#     merged_df['date'] = pd.to_datetime(merged_df['valid_time'])
#     merged_df['year'] = merged_df['date'].dt.year
#     merged_df[variable] = temp_c.values
#     merged_df['month_day'] = merged_df['valid_time'].dt.strftime('%m-%d')
#     print("ANOM NORMALIZED:")
#     print(merged_df)
#     #baseline_df = baseline.to_dataframe().reset_index()
#     # baseline_df = baseline_df.rename(columns={variable: 'baseline'})
#     # baseline_df_unique = baseline_df[['month', 'baseline']].drop_duplicates(subset=[time_step])
#     # df_with_baseline = df.merge(baseline_df_unique[[time_step, 'baseline']], on=time_step, how='left')
#     clean_df = merged_df[['month_day', 'year', variable, 'anom', 'baseline']]
#     return clean_df


def clean_to_df(anom, raw_temp, baseline, variable, weekly=False):
    print('Cleaning data')
    time_step = 'week' if weekly else 'month_day'
    anom.name = 'anom'
    df = anom.to_dataframe().reset_index()
    df['date'] = pd.to_datetime(df['valid_time'])
    df['year'] = df['date'].dt.year
    df[variable] = raw_temp.values
    if weekly:
        df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')
        df = df.groupby([time_step, 'year']).agg({
            'anom': 'mean',
            variable: 'mean',
            'week_start': 'first'
        }).reset_index()
    else:
        df = df.groupby([time_step, 'year'])[['anom', variable]].mean().reset_index()
    baseline_df = baseline.to_dataframe().reset_index()
    baseline_df = baseline_df.rename(columns={variable: 'baseline'})
    baseline_df_unique = baseline_df[[time_step, 'baseline']].drop_duplicates(subset=[time_step])
    df_with_baseline = df.merge(baseline_df_unique[[time_step, 'baseline']], on=time_step, how='left')
    clean_df = df_with_baseline[[time_step, 'year', variable, 'anom', 'baseline']]
    print('Finished cleaning data')

    return clean_df

def pivot_for_plotting_df(clean_df, weekly=False):
    print('Pivoting data for plotting')
    time_step = 'week' if weekly else 'month_day'
    pivot_df = clean_df.pivot(index=time_step, columns='year', values='anom').reset_index()
    pivot_df = pivot_df.merge(clean_df[[time_step, 'baseline']].drop_duplicates(subset=[time_step]), on=time_step, how='left')
    year_strings = clean_df['year'].unique().astype(str).tolist() 
    pivot_df.columns = [time_step] + year_strings + ['baseline']
    print('Finished pivoting data for plotting')
    return pivot_df

def adjust_to_preindustrial_baseline(pivot_df, preindustrial_baseline_adjustment):
    pivot_df['month'] = pivot_df['month_day'].str.split('-').str[0].astype(int)
    # Merge pivot_df_t2m with preindustrial_baseline_adjustment on the month
    merged_df = pd.merge(pivot_df, preindustrial_baseline_adjustment, on='month', how='left')
    years = [col for col in pivot_df_t2m.columns if col not in ['month_day', 'baseline', 'month']]
    for year in years:
        merged_df[year] = merged_df[year] + merged_df['adjustment']
    merged_df = merged_df.drop(columns=['adjustment', 'month'])
    return merged_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pull specified datasets")
    parser.add_argument('--all', action='store_true', help='pull data for all plots')
    parser.add_argument('--ne_sst', action='store_true', help='pull data for northeast_atlantic sst plot')
    parser.add_argument('--sst', action='store_true', help='pull data for global sst plot')
    parser.add_argument('--t2m', action='store_true', help='pull data for global air temperature plot')
    parser.add_argument('--on_t2m', action='store_true', help='pull data for ontario t2m plot')
    parser.add_argument('--ca_t2m', action='store_true', help='pull data for Canada t2m plot')
    parser.add_argument('--north_t2m', action='store_true', help='pull data for northern t2m plot')
    parser.add_argument('--in_dir', type=str, help='directory to read data from', default='data/global_temperatures/era5')
    parser.add_argument('--out_dir', type=str, help='directory to save data to', default='data/global_temperatures/preprocessed_for_plots')
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir

    if args.all or args.ne_sst:
        variable = 'sst'
        print(f'\nProcessing northeast atlantic {variable} data')
        pattern = os.path.join(in_dir, f'era5_northeast_atlantic_{variable}_*.nc')
        files = glob.glob(pattern)
        if len(files) == 1:
            ne_sst_data = xr.open_dataset(files[0]).load()
        else:
            print(f"Expected exactly one file, but found {len(files)} files.")
        preprocessed_ne_sst = preprocess(ne_sst_data, variable)
        #preprocessed_ne_sst.to_csv(os.path.join(out_dir, f'era5_ne_atlantic_{variable}_anom_check.csv'), index=False)
        pivot_df_ne_sst = pivot_for_plotting_df(preprocessed_ne_sst)
        pivot_df_ne_sst.to_csv(os.path.join(out_dir, f'era5_ne_atlantic_{variable}_anom.csv'), index=False)

    if args.all or args.sst:
        variable = 'sst'
        print(f'\nProcessing global {variable} data')
        pattern = os.path.join(in_dir, f'era5_non_polar_seas_{variable}_*.nc')
        files = glob.glob(pattern)
        if files:
            sst_data = xr.open_dataset(files[0]).load()
        else:
            print(f"Expected exactly one file, but found {len(files)} files.")
        preprocessed_sst = preprocess(sst_data, variable)
        #preprocessed_sst.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_check.csv'), index=False)
        pivot_df_sst = pivot_for_plotting_df(preprocessed_sst)
        pivot_df_sst.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_1979-2000_baseline.csv'), index=False)

    if args.all or args.t2m:
        variable = 't2m'
        print(f'\nProcessing global {variable} data')
        pattern = os.path.join(in_dir, f'era5_global_coords_{variable}_*.nc')
        files = glob.glob(pattern)
        #baseline = pd.read_csv('data/global_temperatures/preprocessed_for_plots/pre_industrial_baseline_temps.csv')
        if files:
            t2m_data = xr.open_dataset(files[0]).load()
        else:
            print(f"Expected exactly one file, but found {len(files)} files.")
        preprocessed_t2m = preprocess(t2m_data, variable, weekly=False)
        #preprocessed_t2m.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_weekly_check.csv'), index=False)
        pivot_df_t2m = pivot_for_plotting_df(preprocessed_t2m, weekly=False)
        # https://climate.copernicus.eu/tracking-breaches-150c-global-warming-threshold
        preindustrial_baseline_adjustment = pd.read_csv('data/global_temperatures/preprocessed_for_plots/pre_industrial_baseline_adjustment.csv')
        print(pivot_df_t2m)
        pivot_df_t2m_preindustrial_baseline_adjusted = adjust_to_preindustrial_baseline(pivot_df_t2m, preindustrial_baseline_adjustment)
        print(pivot_df_t2m_preindustrial_baseline_adjusted)
        pivot_df_t2m_preindustrial_baseline_adjusted.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_preindustrial_baseline.csv'), index=False)

    if args.all or args.on_t2m:
        variable = 't2m'
        print(f'\nProcessing Ontario {variable} data')
        pattern = os.path.join(in_dir, f'era5_ontario_{variable}_*.nc')
        files = glob.glob(pattern)
        if files:
            on_t2m_data = xr.open_dataset(files[0]).load()
        else:
            print(f"Expected exactly one file, but found {len(files)} files.")
        preprocessed_t2m = preprocess(on_t2m_data, variable, weekly=False)
        #preprocessed_t2m.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_weekly_check.csv'), index=False)
        pivot_df_t2m = pivot_for_plotting_df(preprocessed_t2m, weekly=False)
        pivot_df_t2m.to_csv(os.path.join(out_dir, f'era5_ontario_{variable}_anom.csv'), index=False)

    if args.all or args.ca_t2m:
        variable = 't2m'
        print(f'\nProcessing Canada {variable} data')
        pattern = os.path.join(in_dir, f'era5_canada_{variable}_*.nc')
        files = glob.glob(pattern)
        if files:
            ca_t2m_data = xr.open_dataset(files[0]).load()
        else:
            print(f"Expected exactly one file, but found {len(files)} files.")
        preprocessed_t2m = preprocess(ca_t2m_data, variable, weekly=False)
        #preprocessed_t2m.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_weekly_check.csv'), index=False)
        pivot_df_t2m = pivot_for_plotting_df(preprocessed_t2m, weekly=False)
        pivot_df_t2m.to_csv(os.path.join(out_dir, f'era5_canada_{variable}_anom.csv'), index=False)

    if args.all or args.north_t2m:
        variable = 't2m'
        print(f'\nProcessing Northern {variable} data')
        pattern = os.path.join(in_dir, f'era5_northern_latitudes_{variable}_*.nc')
        files = glob.glob(pattern)
        if files:
            ca_t2m_data = xr.open_dataset(files[0]).load()
        else:
            print(f"Expected exactly one file, but found {len(files)} files.")
        preprocessed_t2m = preprocess(ca_t2m_data, variable, weekly=False)
        #preprocessed_t2m.to_csv(os.path.join(out_dir, f'era5_{variable}_anom_weekly_check.csv'), index=False)
        pivot_df_t2m = pivot_for_plotting_df(preprocessed_t2m, weekly=False)
        pivot_df_t2m.to_csv(os.path.join(out_dir, f'era5_north_{variable}_anom.csv'), index=False)