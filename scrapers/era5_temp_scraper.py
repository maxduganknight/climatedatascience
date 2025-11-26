import pandas as pd
import numpy as np
import xarray as xr
import cdsapi 
import glob
import sys
import os
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import argparse
import re
import warnings

sys.path.append('/Users/max/Deep_Sky')
from creds import CDS_UID, CDS_API_key

c = cdsapi.Client()

warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed")
warnings.filterwarnings("ignore", message="The specified chunks separate the stored chunks along dimension")

def get_existing_file(base_name):
    files = glob.glob(f'{out_dir}/{base_name}*.nc')
    if len(files) == 1:
        print(f'Found file: {files[0]}.\nWill pull era5 data from the end of this file until the latest.')
        #print(files[0])
        with xr.open_dataset(files[0]) as file:
            return file.load()  # Load data into memory and close the file
    else:
        print(f'More than 1 files or no files found for {out_dir}{base_name}. There should only be one.')
        print(files)
        return None

def get_most_recent_date(file):
    most_recent_date = pd.to_datetime(file.valid_time[-1].values).date()
    return most_recent_date

def pull_era5_daily(variable, existing_file, start_date, end_date, lat_lon_dimensions, out_dir):
    print('Pulling data from CDS API...')
    dataset = "reanalysis-era5-single-levels"
    request = {
        'product_type': ['reanalysis'],
        'variable': variables_dict[variable],
        'year': [str(i) for i in range(start_date.year, end_date.year + 1)],
        'month': [f'{i:02d}' for i in range(1, 13)],
        'day': [f'{i:02d}' for i in range(1, 32)],
        'time': ['00:00'],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
        'area': coordinates_dict[lat_lon_dimensions]
    }
    small_file = f"era5_{lat_lon_dimensions}_{variable}_{end_date.strftime('%Y%m%d')}.nc"
    big_file = f'{out_dir}/temp_{small_file}'
    c = cdsapi.Client()
    c.retrieve(dataset, request, big_file)
    
    with xr.open_dataset(big_file, chunks={'time': 1, 'latitude': 100, 'longitude': 100}) as data:
        temp = data[variable]
        weights = np.cos(np.deg2rad(temp.latitude))
        weights.name = "weights"
        temp_weighted = temp.weighted(weights)
        temp_mean = temp_weighted.mean(["longitude", "latitude"])
        temp_mean_ds = temp_mean.to_dataset(name=variable)
        
    # Drop the 'expver' coordinate if it exists
    if 'expver' in existing_file.coords:
        existing_file = existing_file.drop_vars('expver')
    if 'expver' in temp_mean_ds.coords:
        temp_mean_ds = temp_mean_ds.drop_vars('expver')
    
    combined_data = xr.concat([existing_file, temp_mean_ds], dim='valid_time')
    combined_data = combined_data.sortby('valid_time').drop_duplicates('valid_time', keep='last')
    outfile = os.path.join(out_dir, small_file)
    combined_data.to_netcdf(outfile)
    print(f"Data written to {outfile}.\n")
    os.remove(big_file)
    
# def pull_era5_monthly(variable, start_date, end_date, lat_lon_dimensions, base_name):
#     dataset = "reanalysis-era5-single-levels-monthly-means"
#     request = {
#         'product_type': ['monthly_averaged_reanalysis'],
#         'variable': variable,
#         'year': [str(i) for i in range(start_date.year, end_date.year + 1)],
#         'month': [f'{i:02d}' for i in range(1, 13)],
#         'time': ['00:00'],
#         'data_format': 'netcdf',
#         'download_format': 'unarchived',
#         'area': lat_lon_dimensions
#     }
#     client = cdsapi.Client()
#     out_file = generate_filename(base_name, start_date, end_date)
#     client.retrieve(dataset, request, f'{out_dir}/{out_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pull specified datasets")
    parser.add_argument('--all', action='store_true', help='pull data for all plots')
    parser.add_argument('--ne_sst', action='store_true', help='pull data for northeast_atlantic sst plot')
    parser.add_argument('--sst', action='store_true', help='pull data for global sst plot')
    parser.add_argument('--t2m', action='store_true', help='pull data for global air temperature plot')
    parser.add_argument('--out_dir', type=str, help='directory to save data to', default='data/global_temperatures/era5/')
    args = parser.parse_args()

    today = datetime.datetime.today()
    start_date = datetime.datetime(1979, 1, 1)

    out_dir = args.out_dir

    coordinates_dict = {
        "global_coords": [90, 180, -90, -180],
        "non_polar_seas": [60, 180, -60, -180],
        "northeast_atlantic": [60, -40, 0, 0]
    }

    variables_dict = {
        "sst": "sea_surface_temperature",
        "t2m": "2m_temperature"
    }
    
    if args.all or args.ne_sst:
        variable = 'sst'
        coords = 'northeast_atlantic'
        print(f'Running northeast atlantic {variable} data')
        existing_file = get_existing_file(f"era5_{coords}_{variable}")
        most_recent_date = get_most_recent_date(existing_file)
        pull_era5_daily(variable, existing_file, most_recent_date, today, coords, out_dir)

    if args.all or args.sst:
        variable = 'sst'
        coords = 'non_polar_seas'
        print(f'Running global {variable} data\n')
        existing_file = get_existing_file(f"era5_{coords}_{variable}")
        if existing_file:
            most_recent_date = get_most_recent_date(existing_file)
        else:
            most_recent_date = start_date
        pull_era5_daily(variable, existing_file, most_recent_date, today, coords, out_dir)

    if args.all or args.t2m:
        variable = 't2m'
        coords = 'global_coords'
        print(f'Running global {variable} data\n')
        existing_file = get_existing_file(f"era5_{coords}_{variable}")
        most_recent_date = get_most_recent_date(existing_file)
        pull_era5_daily(variable, existing_file, most_recent_date, today, coords, out_dir)