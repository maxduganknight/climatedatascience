import sys
import os
from pathlib import Path

if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    dashboard_dir = Path(__file__).parent.parent
    sys.path.append(str(dashboard_dir))

import pandas as pd
import numpy as np
import datetime
from utils.logging_utils import setup_logging
from utils.retrieval_utils import load_config, needs_update, save_dataset

logger = setup_logging()

def download_co2_ppm(config: dict) -> pd.DataFrame:
    """
    Download CO2 PPM data using URL from config
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing URLs
        
    Returns
    -------
    pd.DataFrame
        Processed CO2 PPM data
    """
    # MDK we can go back to using NOAA data once they put the Mauna Loa data back online. For now, we'll use the Scripps data.

    # url = config['co2_ppm']['data_source_url_noaa']
    # df = pd.read_csv(url, skiprows=40)
    # columns_of_interest = ['year', 'month', 'decimal date', 'average', 'deseasonalized']
    # df = df[columns_of_interest]
    # df.rename(columns={'average': 'co2_ppm'}, inplace=True)

    url = config['co2_ppm']['data_source_url']
    df = pd.read_csv(url, 
                    skiprows=72,  
                    names=['year', 'month', 'date_excel', 'decimal_date', 'co2_ppm', 
                            'deseasonalized', 'fit', 'seasonally_adjusted_fit',
                            'co2_filled', 'seasonally_adjusted_filled', 'station'],
                    na_values='-99.99')  
    df = df[['year', 'month', 'decimal_date','co2_ppm', 'deseasonalized']].copy()
    
    # Create datetime column and clean up
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.drop(columns=['year', 'month', 'decimal_date'])    
    df = df.dropna(subset=['co2_ppm'])
    df = df.rename(columns={'date': 'year'})
    logger.info(f"Downloaded and processed {len(df)} rows of CO2 PPM data")
    
    return df

def process_co2_ppm(config: dict, is_local: bool = True) -> str:
    """
    Download, process and save CO2 PPM data
    
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
    df = download_co2_ppm(config)
    return save_dataset(df, 'co2_ppm', is_local)

def main():
    """Run data retrieval"""
    config = load_config()
    
    if needs_update('co2_ppm', config):
        output_path = process_co2_ppm(config)
        logger.info(f'Saved processed CO2 PPM data to {output_path}')
    else:
        logger.info("CO2 PPM data is up to date")

if __name__ == "__main__":
    main()