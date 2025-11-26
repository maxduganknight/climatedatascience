import json
import datetime
from pathlib import Path
import pandas as pd
import boto3
import os
import glob
from io import StringIO
import sys
from .paths import CONFIG_PATH, PROCESSED_DIR, PROJECT_ROOT, DATA_DIR
from .logging_utils import setup_logging
import yaml
from typing import Optional

# Replace the default logger with our custom one
logger = setup_logging()

def get_aws_secret():
    """
    Get credentials from AWS Secrets Manager.
    Only called when running in Lambda environment.
    
    Returns
    -------
    dict
        Dictionary of credentials from AWS Secrets Manager
    """
    secret_name = f"dashboard-creds-{os.environ['ENVIRONMENT']}"
    session = boto3.session.Session()
    client = session.client('secretsmanager')
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        logger.error(f"Error getting secret from AWS: {e}")
        raise
    else:
        return json.loads(get_secret_value_response['SecretString'])

def load_config():
    """Load configuration from dataset_dir.json"""
    try:
        logger.info(f"Loading config from: {CONFIG_PATH}")
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        logger.info(f"Config loaded successfully: {list(config.keys())}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def get_files_list(dataset_name: str, is_local: bool = True) -> list:
    """
    Get list of existing files for a dataset from either local filesystem or S3
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    is_local : bool
        Whether to check local files or S3
        
    Returns
    -------
    list
        List of file paths/keys found
    """
    try:
        if is_local:
            pattern = PROCESSED_DIR / f'{dataset_name}_*.csv'
            files = glob.glob(str(pattern))
        else:
            s3 = boto3.client('s3')
            bucket = str(DATA_DIR)
            logger.info(f"Using S3 bucket: {bucket}")
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=f'processed/{dataset_name}_'
            )
            files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
        
        logger.info(f"Found {len(files)} existing files for {dataset_name}")
        return files
        
    except Exception as e:
        logger.error(f"Error getting file list for {dataset_name}: {e}")
        return []

def get_file_date(dataset_name: str, is_local: bool = True) -> Optional[datetime.date]:
    """
    Get the date from the most recent file for a dataset
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    is_local : bool
        Whether to check local files or S3
        
    Returns
    -------
    datetime.date or None
        Date from filename if found, None otherwise
    """
    files = get_files_list(dataset_name, is_local)
    
    if not files:
        return None
        
    if len(files) > 1:
        file_list = "\n".join(files)
        raise RuntimeError(
            f"Multiple files found for dataset {dataset_name}. Expected only one.\n"
            f"Files found:\n{file_list}\n"
            f"Please remove extra files and keep only the most recent."
        )
    
    # Extract date from single file
    try:
        date_str = files[0].split('_')[-1].split('.')[0]  # Extract YYYYMMDD
        return datetime.datetime.strptime(date_str, '%Y%m%d').date()
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid date format in filename: {files[0]}. Expected YYYYMMDD.") from e

def needs_update(dataset_name: str, config: dict, is_local: bool = True) -> bool:
    """
    Check if a dataset needs to be updated based on its most recent file date
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    config : dict
        Configuration dictionary containing update cadence
    is_local : bool
        Whether to check local files or S3
        
    Returns
    -------
    bool
        True if dataset needs update, False otherwise
    """
    logger.info(f"Checking updates for {dataset_name} (is_local={is_local})")
    
    try:
        latest_date = get_file_date(dataset_name, is_local)
        
        # If no files exist, definitely need update
        if latest_date is None:
            logger.info(f"No existing files found for {dataset_name}, update needed")
            return True
        
        today = datetime.date.today()
        days_since_update = (today - latest_date).days
        
        # Get update cadence from config
        cadence = config[dataset_name]['update_cadence']
        
        # Convert cadence to days
        cadence_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'annually': 365
        }
        
        update_interval = cadence_days.get(cadence, 30)  # Default to monthly if cadence not recognized
        needs_update = days_since_update >= update_interval
        
        logger.info(f"Last update: {latest_date}, Days since update: {days_since_update}, "
                   f"Update interval: {update_interval}, Needs update: {needs_update}")
        
        return needs_update
        
    except Exception as e:
        logger.error(f"Error checking updates for {dataset_name}: {e}")
        return True  # Default to update if there's an error

def cleanup_old_files(pattern: str, keep_file: str, is_local: bool = True) -> None:
    """
    Remove old files matching pattern except for the specified keep_file
    
    Parameters
    ----------
    pattern : str
        Glob pattern for files to check
    keep_file : str
        Full path of file to keep
    is_local : bool
        Whether to clean local files or S3
    """
    if is_local:
        # Clean local files
        for file in glob.glob(pattern):
            if file != str(keep_file):
                try:
                    Path(file).unlink()
                    logger.info(f"Removed old file: {file}")
                except Exception as e:
                    logger.error(f"Error removing file {file}: {e}")
    else:
        try:
            s3 = boto3.client('s3')
            bucket = str(DATA_DIR)
            
            # Extract the directory and filename from the keep_file
            keep_dir, keep_filename = os.path.split(keep_file)
            base_name = '_'.join(keep_filename.split('_')[:-1])
            
            # List objects in the bucket with the given prefix
            prefix = f"{keep_dir}/" if keep_dir else ""
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Only process files that match the base name but aren't the keep file
                    if base_name in key and key != keep_file:
                        try:
                            s3.delete_object(Bucket=bucket, Key=key)
                            logger.info(f"Removed old S3 file: {key}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old S3 file {key}: {e}")
        except Exception as e:
            logger.error(f"Error during S3 cleanup: {str(e)}")
            raise

def decimal_year_to_datetime(decimal_year):
    """
    Convert decimal year to datetime object
    
    Parameters
    ----------
    decimal_year : float
        Year with decimal fraction (e.g., 1993.4583)
        
    Returns
    -------
    datetime
        Datetime object
    """
    year = int(decimal_year)
    remainder = decimal_year - year
    days_in_year = 366 if year % 4 == 0 else 365
    days = int(remainder * days_in_year)
    
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=days)

def save_dataset(df: pd.DataFrame, dataset_name: str, is_local: bool = True) -> str:
    """
    Save dataset to either local filesystem or S3
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    dataset_name : str
        Name of dataset (e.g., 'aviso_slr')
    is_local : bool
        Whether to save locally or to S3
        
    Returns
    -------
    str
        Path or S3 URI of saved file
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    
    if is_local:
        output_file = PROCESSED_DIR / f'{dataset_name}_{timestamp}.csv'
        df.to_csv(output_file)
        
        # Cleanup old files
        cleanup_old_files(str(PROCESSED_DIR / f'{dataset_name}_*.csv'), output_file)
        logger.info(f'Saved {dataset_name} data locally to {output_file}')
        return str(output_file)
    else:
        try:
            s3 = boto3.client('s3')
            key = f"processed/{dataset_name}_{timestamp}.csv"
            bucket = str(DATA_DIR)

            # Convert DataFrame to CSV string
            csv_buffer = StringIO()
            df.to_csv(csv_buffer)
            csv_content = csv_buffer.getvalue()
            
            logger.info(f"🔄 Writing {dataset_name} to S3: bucket={bucket}, key={key}")
            
            # Upload to S3 with explicit response checking
            response = s3.put_object(
                Bucket=bucket,
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
            
            s3_path = f"s3://{bucket}/{key}"
            logger.info(f'✅ Saved {dataset_name} data to {s3_path}')
            return s3_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save {dataset_name} to S3: {str(e)}")
            raise