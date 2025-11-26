import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Add the parent directory to the path to allow importing modules
cdr_dashboard_dir = Path(__file__).parent.parent
if str(cdr_dashboard_dir) not in sys.path:
    sys.path.append(str(cdr_dashboard_dir))

# Import utilities
from utils.retrieval_utils import set_api_token, pull_full_table, prepare_dataframe_for_saving
from utils.paths import RAW_DIR
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging()

def retrieve_all_data(api_token=None):
    """
    Pull complete data from CDR.fyi API for all tables and save as CSV files.
    
    Parameters:
    - api_token: CDR.fyi API token (if None, will try to get from creds.py)
    
    Returns:
    - dict: Dictionary of DataFrames with the retrieved data
    """
    # Ensure the raw data directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle API token
    if api_token is None:
        try:
            # Try to import from local creds
            sys.path.append('/Users/max/Deep_Sky/')
            from creds import CDR_FYI_API_TOKEN
            api_token = CDR_FYI_API_TOKEN
            logger.info("Using API token from creds.py")
        except ImportError:
            raise ValueError(
                "API token not provided and could not import from creds.py. "
                "Please provide the token as an argument."
            )
    
    # Set up the API token
    set_api_token(api_token)
    
    # Tables to retrieve
    tables = ['suppliers', 'purchasers', 'orders', 'marketplaces']
    data_frames = {}
    
    # Pull each table
    for table in tables:
        logger.info(f"Retrieving all {table} data from CDR.fyi API...")
        
        try:
            # Get data with index already reset
            df = pull_full_table(table, reset_before_return=True)
            
            if df is not None and not df.empty:
                logger.success(f"Successfully retrieved {len(df)} {table} records")
                
                # No need for prepare_dataframe_for_saving here, already reset
                output_path = RAW_DIR / f"cdr_fyi_{table}.csv"
                df.to_csv(output_path, index=False)
                logger.success(f"Saved {table} data to {output_path}")
                
                # Store in return dictionary
                data_frames[table] = df
            else:
                logger.error(f"Failed to retrieve {table} data - empty or None result")
        
        except Exception as e:
            logger.error(f"Error retrieving {table} data: {str(e)}", exc_info=True)
    
    return data_frames

def main():
    parser = argparse.ArgumentParser(description='Retrieve all data from CDR.fyi API')
    parser.add_argument('--token', help='CDR.fyi API token (if not provided, will try to get from creds.py)')
    args = parser.parse_args()
    
    logger.info("Starting one-time retrieval of all CDR.fyi data...")
    
    try:
        data = retrieve_all_data(api_token=args.token)
        
        # Print summary
        logger.info("=== Retrieval Summary ===")
        for table, df in data.items():
            logger.info(f"{table}: {len(df)} records")
        
        logger.success("Data retrieval complete! CSV files saved to the raw directory.")
    except Exception as e:
        logger.error(f"Data retrieval failed: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())