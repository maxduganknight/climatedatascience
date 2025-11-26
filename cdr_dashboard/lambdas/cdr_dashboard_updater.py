import sys
from pathlib import Path
# Add dashboard directory to Python path
cdr_dashboard_dir = Path(__file__).parent.parent
if str(cdr_dashboard_dir) not in sys.path:
    sys.path.append(str(cdr_dashboard_dir))

import os
import boto3
import json
from datetime import datetime
import pandas as pd
from botocore.exceptions import ClientError
import importlib

# Import utils
from utils.logging_utils import setup_logging
from utils.paths import RAW_DIR, PROCESSED_DIR, LOGS_DIR, IS_LAMBDA, PROJECT_ROOT
from utils.retrieval_utils import save_dataset, prepare_dataframe_for_saving
# Import processor functions
from retrieval.cdr_fyi_processor import (
    process_orders_data,
    process_latest_deals,
    process_purchasers_data,
    process_suppliers_data,
    scrape_dollars_spent,
    create_dollars_df
)
from retrieval.carbon_pricing_processor import (
    process_carbon_pricing_data
)

# Setup logger
logger = setup_logging()

# --- Create a mapping from function names (strings) to actual functions ---
PROCESSOR_FUNCTIONS = {
    "process_orders_data": process_orders_data,
    "process_latest_deals": process_latest_deals,
    "process_purchasers_data": process_purchasers_data,
    "process_suppliers_data": process_suppliers_data,
    "scrape_dollars_spent": scrape_dollars_spent,
    "create_dollars_df": create_dollars_df,
    "process_carbon_pricing_data": process_carbon_pricing_data
}

# --- Function to load the JSON config ---
def load_dataset_config(config_path=None):
    """Loads the dataset configuration from a JSON file."""
    if config_path is None:
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

def get_secret(secret_name, region_name="ca-central-1"):
    """Retrieve a secret from AWS Secrets Manager"""
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        raise
    else:
        return json.loads(get_secret_value_response['SecretString'])

def upload_to_s3(file_path, bucket_name=None, subdirectory='raw'):
    """
    Upload a file to S3
    
    Parameters:
    - file_path: Path to file to upload
    - bucket_name: S3 bucket name (if None, uses environment variable)
    - subdirectory: S3 subdirectory ('raw' or 'processed')
    """
    try:
        s3 = boto3.client('s3')
        if not bucket_name:
            bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
        
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
        
        # Determine the S3 key based on the file path and subdirectory
        file_name = os.path.basename(file_path)
        s3_key = f"{subdirectory}/{file_name}"
        
        logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
        s3.upload_file(
            file_path,
            bucket_name,
            s3_key
        )
        logger.success(f"Successfully uploaded {file_name} to S3")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to S3: {str(e)}", exc_info=True)
        return False

def retrieve_and_process_cdr_data(is_local=False):
    """Retrieve and process all required dashboard data sources."""
    # Create paths based on environment
    if is_local:
        raw_dir = RAW_DIR
        processed_dir = PROCESSED_DIR
    else:
        raw_dir = Path('/tmp/raw')
        processed_dir = Path('/tmp/processed')

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Load environment, utils, API token ---
        environment = os.environ.get('ENVIRONMENT', 'test')
        logger.info(f"Running in {environment} environment")
        bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
        if not bucket_name and not is_local:
             raise ValueError("S3 bucket name not provided in environment variables")

        import utils.retrieval_utils as utils
        importlib.reload(utils)
        from utils.retrieval_utils import set_api_token, update_checker, pull_full_table

        utils.RAW_DIR = str(raw_dir)
        utils.PROCESSED_DIR = str(processed_dir)

        # Get and set API token (logic remains the same)
        # ... (copy the API token retrieval logic here) ...
        if is_local:
            try:
                sys.path.append('/Users/max/Deep_Sky/')
                from creds import CDR_FYI_API_TOKEN
                api_token = CDR_FYI_API_TOKEN
            except ImportError:
                logger.warning("Could not import CDR_FYI_API_TOKEN from creds, trying environment")
                api_token = os.environ.get("CDR_FYI_API_TOKEN")
        else:
            logger.info(f"Getting API token from AWS Secrets Manager")
            secrets = get_secret(f"dashboard-creds-{environment}")
            api_token = secrets.get('CDR_FYI_API_TOKEN')
            logger.info(f"Retrieved API token from secrets (length: {len(api_token) if api_token else 0})")

        if not api_token:
            raise ValueError("CDR.FYI API token not found")
        logger.info("Setting up API token and URL")
        set_api_token(api_token)
        utils.BASE_URL = 'https://api.cdr.fyi/v1'
        # --- End API Token Setup ---

        # --- Load Dataset Configuration ---
        datasets_config = load_dataset_config() # Load from JSON file

        # --- Step 1: Ensure Raw Files Exist & Check for API Updates ---
        logger.info("Ensuring raw data files are available...")
        raw_dataframes = {} # Store loaded raw dataframes
        updated_api_tables = []

        for config in datasets_config:
            raw_filename = config.get('raw_filename')
            if not raw_filename: # Skip datasets without a primary raw file like 'latest_deals'
                continue

            api_table = config.get('api_table')
            file_path = raw_dir / raw_filename
            s3_key = f"raw/{raw_filename}"

            # 1a. Check local existence
            if not file_path.exists():
                logger.info(f"Local file {file_path} not found.")
                downloaded_from_s3 = False
                # 1b. If in Lambda, try downloading from S3
                if not is_local:
                    try:
                        logger.info(f"Attempting to download {s3_key} from S3 bucket {bucket_name}...")
                        s3 = boto3.client('s3')
                        s3.download_file(bucket_name, s3_key, str(file_path))
                        logger.success(f"Successfully downloaded {s3_key} to {file_path}")
                        downloaded_from_s3 = True
                    except ClientError as e:
                        if e.response['Error']['Code'] == '404':
                            logger.warning(f"File {s3_key} not found in S3.")
                        else:
                            logger.error(f"Error downloading {s3_key} from S3: {e}")
                            # If it's a required file (like CCN) and download fails, raise error
                            if not api_table: # Files not from API must exist in S3
                                raise FileNotFoundError(f"Required file {s3_key} missing from S3 and cannot be pulled.") from e

                # 1c. If file still doesn't exist AND it's an API table, pull fresh
                if not file_path.exists() and api_table:
                    logger.info(f"Pulling fresh data for {api_table}...")
                    # Get data with index already reset
                    df = pull_full_table(api_table, reset_before_return=True)
                    df.to_csv(file_path, index=False)
                    logger.info(f"Created {file_path} with {len(df)} records")
                    if not is_local:
                        upload_to_s3(str(file_path), bucket_name=bucket_name, subdirectory='raw')
                elif not file_path.exists() and not api_table:
                     # If not an API table and still doesn't exist (local or failed S3 download)
                     raise FileNotFoundError(f"Required raw file {raw_filename} could not be found locally or downloaded.")

            # 1d. If file exists, check for API updates (only if it's an API table)
            if file_path.exists() and api_table:
                 try:
                     logger.info(f"Checking for updates via API for {api_table}...")
                     # update_checker should handle reading the local file and appending
                     new_rows = update_checker(api_table) # Assumes update_checker uses utils.RAW_DIR
                     if not new_rows.empty:
                         updated_api_tables.append(api_table)
                         logger.success(f"Found and appended {len(new_rows)} new rows for {api_table}")
                         if not is_local:
                             upload_to_s3(str(file_path), bucket_name=bucket_name, subdirectory='raw')
                     else:
                         logger.info(f"No API updates found for {api_table}")
                 except Exception as e:
                     logger.error(f"Error checking for updates in {api_table}: {str(e)}")

            # 1e. Load the raw dataframe into memory if it exists now
            if file_path.exists():
                 logger.info(f"Loading raw data from {file_path}...")
                 try:
                    if file_path.suffix.lower() == '.xlsx':
                        raw_dataframes[config['name']] = pd.read_excel(
                            file_path,
                            sheet_name='Compliance_Gen Info', # MDK this is brittle because it's specific to the carbon pricing file
                            skiprows=1 # MDK this too
                        )
                        
                    else:
                        raw_dataframes[config['name']] = pd.read_csv(
                            file_path,
                            low_memory=False
                        )
                 except Exception as read_err:
                     logger.error(f"Error reading file {file_path}: {read_err}")
                     # Decide how to handle read errors - skip or raise? For now, log and continue.
                     continue # Skip adding this dataframe if read fails
            else:
                 logger.error(f"File {file_path} still not found after retrieval attempts.")


        if updated_api_tables:
            logger.success(f"API updates found and applied for: {', '.join(updated_api_tables)}")
        else:
            logger.info("No API updates found for any table.")


        # --- Step 2: Process Datasets ---
        logger.info("Starting data processing...")

        for config in datasets_config:
            processor_func_name = config.get('processor_func_name')
            if not processor_func_name: # Skip if no processing function name defined
                continue

            # --- Look up the actual function from the name ---
            processor_func = PROCESSOR_FUNCTIONS.get(processor_func_name)
            if not processor_func:
                logger.error(f"Processor function '{processor_func_name}' not found in mapping for dataset '{config['name']}'. Skipping.")
                continue
            # --- End function lookup ---

            output_base = config['output_filename_base']
            required_raw_names = config['requires_raw']
            dataset_name = config['name']

            logger.info(f"Processing dataset: {dataset_name} using {processor_func_name}...")

            # Gather required input dataframes
            input_dfs = []
            missing_raw = False
            for raw_name in required_raw_names:
                if raw_name in raw_dataframes:
                    input_dfs.append(raw_dataframes[raw_name])
                else:
                    logger.error(f"Missing required raw dataframe '{raw_name}' for processing '{dataset_name}'")
                    missing_raw = True
                    break

            if missing_raw:
                continue

            # Call the processor function
            try:
                processed_df = processor_func(*input_dfs)
                # Remove the processed_dir argument from the save_dataset call
                file_path = save_dataset(processed_df, output_base, is_local=is_local, bucket_name=bucket_name)
                logger.success(f"Saved processed {dataset_name} data to {file_path}")
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}", exc_info=True)

        # After processing all datasets, scrape and save the dollars spent value
        try:
            logger.info("Scraping dollars spent from cdr.fyi...")
            dollars_spent = scrape_dollars_spent()
            
            if dollars_spent is not None:
                # Create dollars spent DataFrame
                dollars_df = create_dollars_df(dollars_spent)
                
                # Save using the consistent save_dataset function
                output_path = save_dataset(dollars_df, "dollars_spent", is_local=is_local, bucket_name=bucket_name)
                logger.success(f"Saved dollars spent value (${dollars_spent:,.2f}) to {output_path}")
        except Exception as e:
            logger.error(f"Error scraping or saving dollars spent: {e}", exc_info=True)
        
        logger.success("Data retrieval and processing completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to retrieve and process CDR data: {str(e)}", exc_info=True)
        return False

def lambda_handler(event, context):
    """AWS Lambda handler"""
    logger.info(f"Starting CDR Dashboard Update Lambda")
    if context:
        logger.info(f"Lambda function: {context.function_name}")
        logger.info(f"Request ID: {context.aws_request_id}")
    
    try:
        success = retrieve_and_process_cdr_data(is_local=False)
        
        # Force flush of any pending log messages
        for handler in logger.handlers:
            handler.flush()
        
        if success:
            return {
                'statusCode': 200,
                'body': json.dumps('CDR Dashboard data updated successfully')
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps('Failed to update CDR Dashboard data')
            }
            
    except Exception as e:
        logger.error(f"Lambda handler failed: {str(e)}", exc_info=True)
        # Force flush of logs even on error
        for handler in logger.handlers:
            handler.flush()
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def ensure_local_directories():
    """Ensure required local directories exist"""
    if not IS_LAMBDA: # Use IS_LAMBDA from paths.py
        for path in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        logger.info("Local directories verified")

if __name__ == "__main__":
    ensure_local_directories()
    retrieve_and_process_cdr_data(is_local=True)