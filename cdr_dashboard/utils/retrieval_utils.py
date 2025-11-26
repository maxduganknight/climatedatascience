import re
import argparse
import time
import glob
from io import StringIO
import requests
import pandas as pd
import json
import sys
import os
import boto3
import datetime
from pathlib import Path
from .paths import PROCESSED_DIR, DATA_DIR, RAW_DIR, PROJECT_ROOT
from .logging_utils import setup_logging

# Setup logger
logger = setup_logging()

# Define a variable to hold the API token
CDR_FYI_API_TOKEN = None

# Try to import from creds if running locally
if os.getenv('AWS_LAMBDA_FUNCTION_NAME') is None:
    try:
        sys.path.append('/Users/max/Deep_Sky/')
        from creds import CDR_FYI_API_TOKEN as LOCAL_TOKEN
        CDR_FYI_API_TOKEN = LOCAL_TOKEN
    except ImportError:
        # Will be set later from Lambda environment
        pass

# Define constants needed for API calls
BASE_URL = 'https://api.cdr.fyi/v1'
HEADERS = {
    'x-page': '1',
    'x-limit': '100'
}

# Function to set API token from outside this module
def set_api_token(token):
    global CDR_FYI_API_TOKEN, HEADERS
    CDR_FYI_API_TOKEN = token
    # Update headers with the token
    HEADERS['Authorization'] = f'Bearer {CDR_FYI_API_TOKEN}'

# Initialize headers if token is already available
if CDR_FYI_API_TOKEN:
    HEADERS['Authorization'] = f'Bearer {CDR_FYI_API_TOKEN}'

def load_dataset_config(config_path=None):
    """Loads the dataset configuration from a JSON file."""
    if config_path is None:
        # Use PROJECT_ROOT imported from utils.paths
        config_path = PROJECT_ROOT / 'config' / 'cdr_dataset_dir.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded dataset configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def query_cdr_fyi_api(table, BASE_URL, HEADERS, page=1, limit=100, entity_filter_type=None, entity_filter_id=None):
    """
    Retrieve orders from the API and return as a DataFrame.
    """
    # Check if token is set
    if 'Authorization' not in HEADERS:
        raise ValueError("API token not set. Call set_api_token() before making requests.")
        
    url = "{BASE_URL}/{DATA_TYPE}".format(BASE_URL = BASE_URL, DATA_TYPE = table)
    request_headers = HEADERS.copy()
    request_headers.update({
        "x-page": str(page),
        "x-limit": str(limit)
    })
    params = {}
    if entity_filter_type is not None:
        params["entityFilterType"] = entity_filter_type
    if entity_filter_id is not None:
        params["entityFilterId"] = entity_filter_id
    response = requests.get(url, headers=request_headers, params=params)
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON. Response text:")
        print(response.text)
        return None, False
    df = pd.json_normalize(data[table])
    return df

def pull_full_table(table, set_index=True, reset_before_return=False):
    """
    Pull full tables once to use as running record of what was already recorded.
    
    Parameters:
    - table: Name of table to pull
    - set_index: Whether to set the ID column as index (default: True)
    - reset_before_return: Whether to reset index before returning (default: False)
    
    Returns:
    - DataFrame with data
    """
    all_records = pd.DataFrame()
    id_col = str(table[:-1] + '_id')
    i = 1
    while True:
        records = query_cdr_fyi_api(page=i, table=table, BASE_URL=BASE_URL, HEADERS=HEADERS)
        print('Pulled page: {page_number}'.format(page_number=i))
        if not records.empty:
            if set_index and id_col in records.columns:
                records.set_index(id_col, inplace=True)
            # Exclude empty or all-NA columns
            records = records.dropna(how='all', axis=1)
            all_records = pd.concat([all_records, records])
            i += 1
        else:
            # Reset index if requested before returning
            if reset_before_return and not all_records.empty:
                all_records = all_records.reset_index()
            return all_records

def update_checker(table_name):
    """
    Checks for new rows in the specified table from the CDR_FYI API, 
    updates the local CSV file if new rows are found, and returns the new rows.
    """
    new_rows = False
    old_table_path = os.path.join(RAW_DIR, 'cdr_fyi_{table}.csv'.format(table=table_name))
    
    try:
        # Load existing data
        old_table = pd.read_csv(old_table_path)
        id_col = f"{table_name[:-1]}_id"
        
        # Get old IDs for comparison
        if id_col not in old_table.columns:
            logger.warning(f"Could not find ID column {id_col} in {old_table_path}")
            old_ids = []
        else:
            old_ids = old_table[id_col].tolist()
        
    except Exception as e:
        logger.error(f"Error reading {old_table_path}: {e}")
        return pd.DataFrame()
        
    # Pull new data with index reset
    print("Retrieving {table} data from CDR.FYI API.".format(table=table_name))
    pulled_table = pull_full_table(table_name, reset_before_return=True)
    
    # Compare IDs
    if id_col in pulled_table.columns and old_ids:
        new_rows = pulled_table[~pulled_table[id_col].isin(old_ids)]
    else:
        new_rows = pulled_table
        
    if len(new_rows) > 0:
        new_rows = new_rows.dropna(how='all', axis=1)
        updated_table = pd.concat([old_table, new_rows])
        updated_table_path = os.path.join(RAW_DIR, 'cdr_fyi_{table}.csv'.format(table=table_name))
        updated_table.to_csv(updated_table_path, index=False)
        print('Updates found and added to {updated_table}\n'.format(updated_table=updated_table_path))
    else:
        updated_table = old_table
        print("No updates found.")
    return new_rows

def cleanup_old_files(pattern, keep_file, is_local=True, bucket_name=None):
    """
    Remove old files matching pattern except for the specified keep_file
    
    Parameters
    ----------
    pattern : str
        Glob pattern for files to check
    keep_file : str or Path
        Full path of file to keep
    is_local : bool
        Whether to clean local files or S3
    bucket_name : str, optional
        S3 bucket name to use (if None, will try environment variable)
    """
    keep_file = str(keep_file)  # Ensure keep_file is a string
    
    if is_local:
        # Clean local files
        for file in glob.glob(pattern):
            if file != keep_file:
                try:
                    Path(file).unlink()
                    logger.info(f"Removed old file: {file}")
                except Exception as e:
                    logger.error(f"Error removing file {file}: {e}")
    else:
        try:
            # Get bucket name from parameter or environment
            if not bucket_name:
                bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
            
            if not bucket_name:
                raise ValueError("S3 bucket name not provided and not available in environment")
                
            s3 = boto3.client('s3')
            
            # Extract the prefix from the pattern
            prefix = pattern.split('*')[0]
            
            # Extract dataset name from keep_file
            # (Assumes pattern like "processed/dataset_name_*.csv")
            dataset_name = os.path.basename(keep_file).split('_')[0]
            if len(os.path.basename(keep_file).split('_')) > 1:
                # Handle multi-word dataset names like "orders_for_viz"
                parts = os.path.basename(keep_file).split('_')
                # Reconstruct the dataset name minus the date part
                dataset_name = '_'.join(parts[:-1])
            
            logger.info(f"Cleaning up old files for dataset: {dataset_name}")
            
            # List objects in the bucket with the given prefix
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Only process files that match the pattern but aren't the keep file
                    if key != keep_file and dataset_name in key:
                        try:
                            s3.delete_object(Bucket=bucket_name, Key=key)
                            logger.info(f"Removed old S3 file: {key}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old S3 file {key}: {e}")
        except Exception as e:
            logger.error(f"Error during S3 cleanup: {str(e)}")
            raise

def save_dataset(df: pd.DataFrame, dataset_name: str, is_local: bool = True, bucket_name: str = None) -> str:
    """
    Save dataset to either local filesystem or S3
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    dataset_name : str
        Name of dataset (e.g., 'orders_for_viz')
    is_local : bool
        Whether to save locally or to S3
    bucket_name : str, optional
        S3 bucket name to use (if None, will try environment variable)
        
    Returns
    -------
    str
        Path or S3 URI of saved file
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    
    if is_local:
        # Make sure PROCESSED_DIR is a Path object
        processed_dir = Path(PROCESSED_DIR) if isinstance(PROCESSED_DIR, str) else PROCESSED_DIR
        output_file = processed_dir / f'{dataset_name}_{timestamp}.csv'
        
        # Ensure the directory exists
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        # Cleanup old files
        pattern = str(processed_dir / f'{dataset_name}_*.csv')
        cleanup_old_files(pattern, output_file)
        logger.info(f'Saved {dataset_name} data locally to {output_file}')
        return str(output_file)
    else:
        try:
            s3 = boto3.client('s3')
            key = f"processed/{dataset_name}_{timestamp}.csv"
            
            # Get bucket name from parameter, environment variable, or default
            if not bucket_name:
                bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
            
            if not bucket_name:
                raise ValueError("S3 bucket name not provided and not available in environment")

            # Convert DataFrame to CSV string
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            logger.info(f"🔄 Writing {dataset_name} to S3: bucket={bucket_name}, key={key}")
            
            # Upload to S3 with explicit response checking
            response = s3.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=csv_content
            )
            
            # Check response
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                logger.info(f"📝 S3 put_object successful, ETag: {response['ETag']}")
            else:
                logger.error(f"❌ S3 put_object failed: {response}")
            
            # Cleanup old files
            logger.info(f"📝 Successfully wrote {dataset_name} to S3, cleaning up old files...")
            pattern = f"processed/{dataset_name}_*.csv"
            cleanup_old_files(pattern, key, is_local=False)
            
            s3_path = f"s3://{bucket_name}/{key}"
            logger.info(f'✅ Saved {dataset_name} data to {s3_path}')
            return s3_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save {dataset_name} to S3: {str(e)}")
            raise

def prepare_dataframe_for_saving(df, ensure_id_column=None):
    """
    Prepare DataFrame for saving to ensure ID columns are preserved.
    
    Parameters:
    - df: DataFrame to prepare
    - ensure_id_column: Optional ID column name to verify exists
    
    Returns:
    - Prepared DataFrame with index reset
    """
    # Reset index to make any index a regular column
    df = df.reset_index()
    
    # If specific ID column is expected, ensure it exists
    if ensure_id_column and ensure_id_column not in df.columns:
        if 'index' in df.columns:
            df.rename(columns={'index': ensure_id_column}, inplace=True)
    
    return df