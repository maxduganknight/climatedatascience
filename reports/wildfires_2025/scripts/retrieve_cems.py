import pandas as pd
import numpy as np
import xarray as xr
import cdsapi 
from datetime import datetime
import glob
import sys
import os
from contextlib import contextmanager
import shutil

# creds
sys.path.append('/Users/max/Deep_Sky')
from creds import CEMS_UID, CEMS_API_KEY

DATADIR = '/Users/max/Deep_Sky/GitHub/Deep_Sky_Data_Science/data'

# API configurations
CDS_CONFIG = {
    'url': 'https://cds.climate.copernicus.eu/api',
    'key': CEMS_API_KEY
}

EWDS_CONFIG = {
    'url': 'https://ewds.climate.copernicus.eu/api',
    'key': CEMS_API_KEY
}

@contextmanager
def temp_api_config(config):
    """Temporarily modify .cdsapirc file for different API endpoints"""
    cdsapirc = os.path.expanduser('~/.cdsapirc')
    backup_file = cdsapirc + '.backup'
    
    # Backup existing config if it exists
    if os.path.exists(cdsapirc):
        shutil.copy2(cdsapirc, backup_file)
    
    # Write new config
    with open(cdsapirc, 'w') as f:
        f.write(f"url: {config['url']}\nkey: {config['key']}")
    
    try:
        yield
    finally:
        # Restore original config
        if os.path.exists(backup_file):
            shutil.move(backup_file, cdsapirc)

# Get current date for file naming
today = datetime.now().strftime('%Y%m%d')

def get_cems_data(region_info, year, month, day):
    """
    Retrieve CEMS fire weather index data for a specified region and time period.
    
    Args:
        region_info (dict): Dictionary containing region name and boundaries
        year (int): Year to retrieve data for
        month (int or list): Month(s) to retrieve data for
        day (int or list): Day(s) to retrieve data for
    
    Returns:
        str: Path to downloaded file
    """
    region_name = region_info['name']
    
    # Create API request
    if year < 2025:
        dataset_type = 'consolidated_dataset'
    else:
        dataset_type = 'intermediate_dataset'

    request = {
        'variable': ['fire_weather_index'],
        'product_type': 'reanalysis',
        'dataset_type': dataset_type,
        'system_version': ['4_1'],
        'year': [str(year)],
        'month': [str(m).zfill(2) for m in month] if isinstance(month, list) else [str(month).zfill(2)],
        'day': [str(d).zfill(2) for d in day] if isinstance(day, list) else [str(day).zfill(2)],
        'grid': '0.25/0.25',
        'data_format': 'netcdf'
    }
    
    # Add area parameter if boundaries are specified
    if all(k in region_info for k in ['north_bound', 'west_bound', 'south_bound', 'east_bound']):
        # Format is: north/west/south/east
        area = [
            region_info['north_bound'],
            region_info['west_bound'],
            region_info['south_bound'],
            region_info['east_bound']
        ]
        request['area'] = '/'.join(map(str, area))
    
    # Create output directory and filename
    output_file = f'{DATADIR}/wildfires/CEMS_fwi_{region_name}_{year}_{today}.nc'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Retrieve data from CEMS API
    print(f"Retrieving data for {region_name}, year {year}...")
    with temp_api_config(EWDS_CONFIG):
        client = cdsapi.Client()
        client.retrieve('cems-fire-historical-v1', request).download(output_file)
    
    print(f"Data saved to {output_file}")
    return output_file

def get_all_days():
    """Return a list of all possible days in a month."""
    return [
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', 
        '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', 
        '25', '26', '27', '28', '29', '30', '31'
    ]

def get_all_months():
    """Return a list of all months in a year."""
    return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

def main():
    """Main function to retrieve CEMS data for defined regions."""
    # Define regions - ONLY EDIT THIS SECTION FOR REUSE
    
    north_america_region = {
        'name': 'north_america',
        'north_bound': 70,
        'south_bound': 14,
        'west_bound': -140,
        'east_bound': -50
    }

    continential_us_region = {
        'name': 'continental_us',
        'north_bound': 49,
        'south_bound': 25,
        'west_bound': -125,
        'east_bound': -66.5
    }

    # US Northeast region
    ne_us_region = {
        'name': 'ne_america',
        'north_bound': 50,
        'south_bound': 40,
        'west_bound': -80,
        'east_bound': -65
    }

    east_coast_region = {
        'name': 'east_coast',
        'north_bound': 45,
        'south_bound': 25,
        'west_bound': -85,
        'east_bound': -65
    }
    
    # Connecticut region
    connecticut_region = {
        'name': 'connecticut',
        'north_bound': 42,
        'south_bound': 41,
        'west_bound': -74,
        'east_bound': -72
    }
    
    # Western US region
    western_us_region = {
        'name': 'western_us',
        'north_bound': 50,
        'south_bound': 24.5,
        'west_bound': -124.4,
        'east_bound': -94.5
    }
    
    # Define regions to process - UNCOMMENT OR ADD AS NEEDED
    regions_to_process = [
        north_america_region
    ]
    
    # Define years to retrieve - adjust as needed
    years_to_retrieve = range(2024, 2026)  # Start year (inclusive) to end year (exclusive)
    
    # Process each region
    for region in regions_to_process:
        for year in years_to_retrieve:
            get_cems_data(
                region_info=region,
                year=year,
                month=get_all_months(),
                day=get_all_days()
            )

if __name__ == '__main__':
    main()
