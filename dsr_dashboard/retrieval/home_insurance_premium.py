import sys
from pathlib import Path
import os

# Only modify path if not running in Lambda
if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    dashboard_dir = Path(__file__).parent.parent
    deep_sky_dir = Path(__file__).parent.parent.parent.parent.parent
    sys.path.append(str(dashboard_dir))
    sys.path.append(str(deep_sky_dir))

import logging
import json
import boto3
import pandas as pd
import numpy as np
import datetime
import requests
from utils.retrieval_utils import load_config, needs_update, save_dataset, get_aws_secret

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_home_insurance_premium(config: dict) -> pd.DataFrame:
    """
    Download home insurance premium data from FRED using their API
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing URLs and series ID
        
    Returns
    -------
    pd.DataFrame
        Processed home insurance premium data
    """
    logger.info("Starting home insurance premium data download")
    
    # Get credentials based on environment
    try:
        if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            secrets = get_aws_secret()
            FRED_API_KEY = secrets['FRED_API_KEY']
        else:
            from creds import FRED_API_KEY

        logger.info("Successfully retrieved API key")
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        raise

    # Construct API URL
    base_url = config['home_insurance_premium']['data_source_url']
    series_id = config['home_insurance_premium']['dataset_id'] 
    
    # Set up parameters for API request
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': '1998-06-01'
    }
    
    try:
        # Make API request
        logger.info("Making API request to FRED")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['observations'])
        
        # Convert date and value columns
        df['date'] = pd.to_datetime(df['date'])
        # Use errors='coerce' to handle missing values (like ".")
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Rename columns to match our standard format
        df = df[['date', 'value']].rename(columns={'value': 'premium_index', 'date': 'year'})
        df['percent_increase'] = round(df['premium_index'] - 100, 2)
        # Remove any rows where value is missing
        df = df.dropna(subset=['percent_increase'])
        
        logger.info(f"Downloaded and processed {len(df)} rows of Home Insurance Premium data")
        logger.info(f"Date range: {df['year'].min()} to {df['year'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading FRED data: {str(e)}")
        raise

def process_home_insurance_premium(config: dict, is_local: bool = True) -> str:
    """
    Download, process and save Home Insurance Premium data
    
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
    logger.info("Starting home insurance premium data processing")
    df = download_home_insurance_premium(config)
    output_path = save_dataset(df, 'home_insurance_premium', is_local)
    logger.info(f"Saved data to {output_path}")
    return output_path

def main():
    """Run data retrieval"""
    logger.info("Starting main execution")
    try:
        config = load_config()
        logger.info("Loaded configuration")
        
        if needs_update('home_insurance_premium', config):
            logger.info("Data needs update, processing...")
            output_path = process_home_insurance_premium(config)
            logger.info(f'Saved processed Home Insurance Premium data to {output_path}')
        else:
            logger.info("Home Insurance Premium data is up to date")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()