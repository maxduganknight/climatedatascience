import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
from datetime import datetime
import glob
import sys
import os
import json
from contextlib import contextmanager
import shutil

sys.path.append('/Users/max/Deep_Sky')
from creds import ECMWF_API_KEY

DATA_DIR = '/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/data'

# Get current date for file naming
today = datetime.now().strftime('%Y%m%d')

def get_precipitation_data(region_info, year):
    """
    Retrieve ECMWF drought index data for a specified region and time period.
    
    Args:
        region_info (dict): Dictionary containing region name and boundaries
        year (int): Year to retrieve data for
        month (int or list): Month(s) to retrieve data for
        day (int or list): Day(s) to retrieve data for
    
    Returns:
        str: Path to downloaded file
    """
    region_name = region_info['name']
    today = datetime.now().strftime('%Y%m%d')
    
    # Create output directory and filename
    output_file = f'{DATADIR}/wildfires/precipitation_{region_name}_{year}_{today}.nc'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create API request for drought indicators
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "data_format": "netcdf",
        "download_format": "zip",
        "variable": ["total_precipitation"],
        "product_type": ["reanalysis"],
        "year": str(year),        
        "month": [
            "01", "02", "03", "04", "05", "06", 
            "07", "08", "09", "10", "11", "12"
        ],
        "time": ["00:00"],
        "area": [
            region_info['north_bound'],
            region_info['west_bound'],
            region_info['south_bound'],
            region_info['east_bound']
        ]
    }
    
    
    # Try different dataset names and endpoints
    dataset = 'reanalysis-era5-single-levels-monthly-means'
    
    print(f"Retrieving data for {region_name}...")
    
    try:
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(output_file)
        print(f"Success! Data saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error with dataset {dataset}: {e}")


if __name__ == '__main__':    
    # Define region boundaries
    region_info = {
        'name': 'contiguous_us',
        'north_bound': 49,
        'west_bound': -125,
        'south_bound': 25,
        'east_bound': -66
    }
    
    for year in range(1975, 2025):
        get_precipitation_data(region_info, year)
    