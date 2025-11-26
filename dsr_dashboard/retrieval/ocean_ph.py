import sys
import os
from pathlib import Path

# Only modify path if not running in Lambda
if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    dashboard_dir = Path(__file__).parent.parent
    sys.path.append(str(dashboard_dir))

import pandas as pd
import numpy as np
import datetime
import logging
import copernicusmarine
import xarray as xr
from utils.logging_utils import setup_logging
from utils.retrieval_utils import load_config, needs_update, save_dataset, get_secret
from utils.era5_utils import create_time_dataframe

# Set up logging
logger = setup_logging()

END_DATE = '20250101'

def ensure_copernicus_login():
    """Ensure we're logged into Copernicus Marine, login only if needed"""
    creds_file = Path.home() / '.copernicusmarine' / '.copernicusmarine-credentials'
    
    if not creds_file.exists() or os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        # Get credentials
        logger.info("Getting credentials...")
        secrets = get_secret()
        COPERNICUS_MARINE_USERNAME = secrets['COPERNICUS_MARINE_USERNAME']
        COPERNICUS_MARINE_PASSWORD = secrets['COPERNICUS_MARINE_PASSWORD']

        # Login to Copernicus Marine
        logger.info("Logging in to Copernicus Marine...")
        try:
            copernicusmarine.login(
                username=COPERNICUS_MARINE_USERNAME, 
                password=COPERNICUS_MARINE_PASSWORD,
                override_credentials=True
            )
        except Exception as e:
            logger.error(f"Error logging in to Copernicus Marine: {e}")
            raise
    else:
        logger.info("Using existing Copernicus Marine credentials")

def download_ocean_ph(config: dict, data_dir = 'data/') -> pd.DataFrame:
    """
    Download global ocean pH data using Copernicus Marine Service
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing credentials
        
    Returns
    -------
    pd.DataFrame
        Processed pH data with columns: year, ph_value, uncertainty
    """
    logger.info("Starting ocean pH data download...")
    
    # Ensure we're logged in
    ensure_copernicus_login()
    
    # Dataset ID for Global Ocean acidification pH time series
    dataset_dir = load_config('dataset_dir.json')
    dataset_id = dataset_dir['ocean_ph']['dataset_id']
    
    # Ensure raw directory exists
    raw_dir = Path(data_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = raw_dir / f'ocean_ph_year_{END_DATE}.nc'
    logger.info(f"Will save raw data to {output_file}")

    try:
        logger.info(f"Downloading data for dataset {dataset_id}...")
        data = copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=["ph"],
            start_datetime="1985-01-01", 
            end_datetime=END_DATE,
            output_filename=output_file
        )
        
        # Read the NetCDF file into a pandas DataFrame
        logger.info("Reading NetCDF file...")
        ds = xr.open_dataset(output_file)
        return ds
        
    except Exception as e:
        logger.error(f"Error downloading ocean pH data: {e}")
        raise

def process_ocean_ph(config: dict, is_local: bool = True) -> str:
    """
    Download, process and save Ocean pH data
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    is_local : bool
        Whether to save locally or to S3
        
    Returns
    -------
    str
        Path or S3 URI of saved file
    """
    ds = download_ocean_ph(config)
    
    # Convert to DataFrame, selecting only time and ph variables
    df = ds['ph'].to_dataframe().reset_index()[['time', 'ph']]
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'time': 'date',
        'ph': 'ph_value'
    })
    
    return save_dataset(df, 'ocean_ph', is_local)

def main():
    """Run data retrieval"""
    config = load_config()
    
    if needs_update('ocean_ph', config):
        output_path = process_ocean_ph(config)
        logging.info(f'Saved processed Ocean pH data to {output_path}')
    else:
        logging.info("Ocean pH data is up to date")

if __name__ == "__main__":
    main()