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
from utils.logging_utils import setup_logging
from utils.retrieval_utils import load_config, needs_update, save_dataset

logger = setup_logging()

def download_noaa_billion(config: dict) -> pd.DataFrame:
    """
    Download NOAA billion-dollar disasters data using URL from config
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing URLs
        
    Returns
    -------
    pd.DataFrame
        Processed disaster data
    """
    url = config['noaa_billion']['data_source_url']
    df = pd.read_csv(url, skiprows=2)
    columns_of_interest = ['Year', 'All Disasters Count', 'All Disasters Cost']
    df = df[columns_of_interest]
    df.rename(columns={
        'Year': 'year',
        'All Disasters Count': 'disaster_count',
        'All Disasters Cost': 'disaster_cost'
    }, inplace=True)
    
    logger.info(f"Downloaded and processed {len(df)} rows of NOAA Billion-Dollar Disaster data")
    logger.info(f"Year range: {df['year'].min()} to {df['year'].max()}")
    
    return df

def process_noaa_billion(config: dict, is_local: bool = True) -> str:
    """
    Download, process and save NOAA Billion-Dollar Disaster data
    
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
    df = download_noaa_billion(config)
    return save_dataset(df, 'noaa_billion', is_local)

def main():
    """Run data retrieval"""
    config = load_config()
    
    if needs_update('noaa_billion', config):
        output_path = process_noaa_billion(config)
        logger.info(f'Saved processed NOAA Billion-Dollar Disaster data to {output_path}')
    else:
        logger.info("NOAA Billion-Dollar Disaster data is up to date")

if __name__ == "__main__":
    main()