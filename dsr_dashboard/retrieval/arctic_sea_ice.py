import sys
import os
from pathlib import Path

# Only modify path if not running in Lambda
if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    dashboard_dir = Path(__file__).parent.parent
    sys.path.append(str(dashboard_dir))

import pandas as pd
import requests
from utils.logging_utils import setup_logging
from utils.retrieval_utils import load_config, needs_update, save_dataset

logger = setup_logging()

def download_arctic_sea_ice(config: dict) -> pd.DataFrame:
    """
    Download and process Arctic Sea Ice Extent data
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing URLs
        
    Returns
    -------
    pd.DataFrame
        Processed sea ice extent data with columns:
        - year (datetime in YYYY-MM-DD format)
        - extent (sea ice extent in million square kilometers)
    """
    url = config['arctic_sea_ice']['data_source_url']
    
    try:
        # Download the CSV file
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download data. Status code: {response.status_code}")
            
        # Convert the CSV data to DataFrame
        df = pd.read_csv(url)

        # Select relevant columns and rename them
        df = df[['year', ' extent']].copy()  # Add .copy() to avoid warning
        df.columns = ['year', 'extent']
        
        # Convert year to datetime format
        df['year'] = pd.to_datetime(df['year'].astype(str) + '-09-01')
        
        logger.info(f"Downloaded and processed {len(df)} rows of Arctic Sea Ice Extent data")
        logger.info(f"Date range: {df['year'].min()} to {df['year'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading Arctic Sea Ice Extent data: {str(e)}")
        raise

def process_arctic_sea_ice(config: dict, is_local: bool = True) -> str:
    """
    Download, process and save Arctic Sea Ice Extent data
    
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
    df = download_arctic_sea_ice(config)
    return save_dataset(df, 'arctic_sea_ice', is_local)

def main():
    """Run data retrieval"""
    config = load_config()
    
    if needs_update('arctic_sea_ice', config):
        output_path = process_arctic_sea_ice(config)
        logger.info(f'Saved processed Arctic Sea Ice Extent data to {output_path}')
    else:
        logger.info("Arctic Sea Ice Extent data is up to date")

if __name__ == "__main__":
    main()