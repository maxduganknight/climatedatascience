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
import requests
from utils.logging_utils import setup_logging
from utils.retrieval_utils import load_config, needs_update, save_dataset, decimal_year_to_datetime

logger = setup_logging()

def download_aviso_slr(config: dict) -> pd.DataFrame:
    """
    Download and process Aviso+ Sea Level Rise data
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing URLs
        
    Returns
    -------
    pd.DataFrame
        Processed sea level rise data with columns:
        - date (datetime in YYYY-MM-DD format)
        - msl (mean sea level in meters)
    """
    url = config['aviso_slr']['data_source_url']
    
    try:
        # Download the text file
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download data. Status code: {response.status_code}")
            
        # Convert the space-separated text data to DataFrame
        data = [line.split() for line in response.text.strip().split('\n')]
        df = pd.DataFrame(data, columns=['time', 'msl'])
        
        # Convert to proper types
        df['time'] = pd.to_numeric(df['time'])
        df['msl'] = pd.to_numeric(df['msl'])
        
        # Convert decimal date to datetime
        df['year'] = df['time'].apply(decimal_year_to_datetime)
        
        # Convert mean sea level meters to mm
        df['msl'] = df['msl'] * 1000

        # Drop the original time column and reorder
        df = df[['year', 'msl']]
        
        logger.info(f"Downloaded and processed {len(df)} rows of Aviso+ Sea Level Rise data")
        logger.info(f"Date range: {df['year'].min()} to {df['year'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading Aviso+ data: {str(e)}")
        raise

def process_aviso_slr(config: dict, is_local: bool = True) -> str:
    """
    Download, process and save Aviso+ Sea Level Rise data
    
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
    df = download_aviso_slr(config)
    return save_dataset(df, 'aviso_slr', is_local)

def main():
    """Run data retrieval"""
    config = load_config()
    
    if needs_update('aviso_slr', config):
        output_path = process_aviso_slr(config)
        logger.info(f'Saved processed Aviso+ Sea Level Rise data to {output_path}')
    else:
        logger.info("Aviso+ Sea Level Rise data is up to date")

if __name__ == "__main__":
    main()