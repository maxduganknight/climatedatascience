import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
from datetime import datetime, date
import glob
import sys
import os
from contextlib import contextmanager
import shutil
import time

# creds
sys.path.append('/Users/max/Deep_Sky')
# from creds import CAMS_UID, CAMS_API_KEY

DATADIR = '/Users/max/Deep_Sky/GitHub/datascience-platform/fundraising_viz/data/wildfire_emissions'
os.makedirs(DATADIR, exist_ok=True)

dataset = "cams-global-fire-emissions-gfas"
client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")

def download_cams_emissions(year):
    """Download CAMS GFAS emissions for a single year"""
    output_file = os.path.join(DATADIR, f'cams_wildfire_emissions_{year}.nc')

    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping...")
        return output_file

    request = {
        "variable": "wildfire_flux_of_carbon_dioxide",
        "date": f"{year}-01-01/{year}-12-31",
        "data_format": "netcdf"
    }

    print(f"Downloading CAMS emissions for {year}...")
    client.retrieve(dataset, request).download(output_file)
    print(f"Downloaded: {output_file}")
    return output_file

def get_missing_years():
    """Identify which years are missing from the data directory"""
    existing_files = glob.glob(os.path.join(DATADIR, 'cams_wildfire_emissions_*.nc'))
    existing_years = set()

    for file in existing_files:
        filename = os.path.basename(file)
        # Extract year from filename like cams_wildfire_emissions_2005.nc
        parts = filename.replace('.nc', '').split('_')
        year = int(parts[-1])
        existing_years.add(year)

    # Generate all years from 2005 to current year
    current_year = date.today().year
    all_years = set(range(2003, current_year + 1))
    missing_years = sorted(all_years - existing_years)

    return missing_years

def main():
    """Download all missing CAMS emissions data one year at a time"""
    missing_years = get_missing_years()

    if not missing_years:
        print("All years are already downloaded!")
        return

    print(f"Missing years: {missing_years}")
    print(f"Will download {len(missing_years)} file(s)")

    for year in missing_years:
        try:
            download_cams_emissions(year)
            # Add delay between requests to be respectful to the API
            time.sleep(10)
        except Exception as e:
            print(f"Error downloading {year}: {e}")
            continue

if __name__ == "__main__":
    main()