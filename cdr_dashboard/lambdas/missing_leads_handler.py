import sys
import os
import json
import boto3
from pathlib import Path
from datetime import datetime
import pandas as pd
from botocore.exceptions import ClientError

# Add dashboard directory to Python path
cdr_dashboard_dir = Path(__file__).parent.parent
if str(cdr_dashboard_dir) not in sys.path:
    sys.path.append(str(cdr_dashboard_dir))

# Import sales module
sys.path.append(str(Path(__file__).parent.parent))
from sales.missing_leads_checker import (
    pull_attio_data,
    process_attio_data,
    retrieve_cdr_fyi_purchasers, 
    missing_leads_checker,
    write_leads_to_google_sheet
)

from utils.logging_utils import setup_logging
from utils.paths import RAW_DIR, PROCESSED_DIR, SALES_DIR, IS_LAMBDA

logger = setup_logging()

def get_secret(secret_name, region_name="ca-central-1"):
    """Retrieve a secret from AWS Secrets Manager"""
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        return json.loads(get_secret_value_response['SecretString'])
    except Exception as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        raise

def upload_to_s3(file_path, bucket_name=None, subdirectory='sales'):
    """Upload a file to S3"""
    try:
        s3 = boto3.client('s3')
        if not bucket_name:
            bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
        
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
        
        file_name = os.path.basename(file_path)
        s3_key = f"{subdirectory}/{file_name}"
        
        logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
        s3.upload_file(file_path, bucket_name, s3_key)
        logger.info(f"Successfully uploaded {file_name} to S3")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to S3: {str(e)}", exc_info=True)
        return False

def ensure_required_files(is_local=False):
    """Ensure that required files exist by downloading from S3 if needed"""
    if is_local:
        # In local mode, files should already be in RAW_DIR
        return True
    
    # In Lambda, files need to be downloaded from S3
    bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
    if not bucket_name:
        logger.error("CDR_DASHBOARD_BUCKET environment variable not set")
        return False
    
    # Create required directories
    for directory in [RAW_DIR, PROCESSED_DIR, SALES_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # List of required files
    required_files = [
        ('raw/cdr_fyi_purchasers.csv', str(RAW_DIR / 'cdr_fyi_purchasers.csv')),
        ('raw/cdr_fyi_orders.csv', str(RAW_DIR / 'cdr_fyi_orders.csv'))
    ]
    
    s3 = boto3.client('s3')
    all_files_exist = True
    
    for s3_key, local_path in required_files:
        try:
            logger.info(f"Downloading {s3_key} from S3 bucket {bucket_name}...")
            s3.download_file(bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded {s3_key} to {local_path}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"File {s3_key} not found in S3 bucket {bucket_name}")
            else:
                logger.error(f"Error downloading {s3_key}: {e}")
            all_files_exist = False
    
    return all_files_exist

def process_missing_leads(is_local=False):
    """Run the missing leads checker process"""
    
    try:
        # In Lambda, we use /tmp directory, for local we use the project structure
        if not is_local:
            raw_dir = Path('/tmp/raw')
            processed_dir = Path('/tmp/processed')
            sales_dir = Path('/tmp/sales')
        else:
            raw_dir = RAW_DIR
            processed_dir = PROCESSED_DIR
            sales_dir = SALES_DIR
        
        # Create necessary directories
        for directory in [raw_dir, processed_dir, sales_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Get environment and credentials
        if is_local:
            # For local testing, use credentials from creds.py
            logger.info("Running in local mode, using credentials from creds.py")
            sys.path.append('/Users/max/Deep_Sky/')
            try:
                from creds import ATTIO_API_TOKEN, GOOGLE_SERVICE_ACCOUNT
                attio_api_token = ATTIO_API_TOKEN
                import sales.missing_leads_checker as mlc
                mlc.GOOGLE_SERVICE_ACCOUNT = GOOGLE_SERVICE_ACCOUNT
            except ImportError:
                logger.error("Could not import credentials from creds.py")
                return {
                    'success': False,
                    'error': "Missing local credentials"
                }
            google_sheet_id = "1t2iux4ZjivaRAGaWaIi1dnCcVGIbkcn_0rQtnFrdKmQ"  # Hardcoded for local testing
        else:
            # For Lambda, get from environment and secrets
            environment = os.environ.get('ENVIRONMENT', 'test')
            logger.info(f"Running in {environment} environment")
            
            # Get API token and Google credentials from secrets
            secrets = get_secret(f"dashboard-creds-{environment}")
            attio_api_token = secrets.get('ATTIO_API_TOKEN')
            google_service_account_json = secrets.get('GOOGLE_SERVICE_ACCOUNT')
            
            # Set the credentials in the missing_leads_checker module
            import sales.missing_leads_checker as mlc
            mlc.ATTIO_API_TOKEN = attio_api_token
            
            # Convert string back to dictionary if needed
            if isinstance(google_service_account_json, str):
                mlc.GOOGLE_SERVICE_ACCOUNT = json.loads(google_service_account_json)
            else:
                mlc.GOOGLE_SERVICE_ACCOUNT = google_service_account_json
            
            # Get sheet ID from environment variable
            google_sheet_id = os.environ.get('GOOGLE_SHEET_ID')
        
        # Ensure required files exist (download from S3 if needed)
        if not is_local:
            logger.info("Ensuring required files exist...")
            if not ensure_required_files(is_local):
                logger.error("Failed to ensure required files exist")
                return {
                    'success': False,
                    'error': "Failed to download required files from S3"
                }
        
        # Define URL for Attio API
        url = "https://api.attio.com/v2"
        
        # Pull data from Attio
        logger.info("Pulling data from Attio API...")
        attio_data = pull_attio_data(url, attio_api_token)
        
        # Process Attio data
        logger.info("Processing Attio data...")
        processed_attio_data = process_attio_data(attio_data)
        
        # Get CDR.fyi purchasers data
        logger.info("Retrieving CDR.fyi purchaser data...")
        purchasers_path = os.path.join(raw_dir, "cdr_fyi_purchasers.csv")
        orders_path = os.path.join(raw_dir, "cdr_fyi_orders.csv")
        cdr_fyi_purchasers = retrieve_cdr_fyi_purchasers(purchasers_path, orders_path)
        
        # Create output file paths
        timestamp = datetime.now().strftime('%Y%m%d')
        missing_leads_path = os.path.join(sales_dir, f"missing_leads_{timestamp}.csv")
        fuzzy_matches_path = os.path.join(sales_dir, f"fuzzy_matches_{timestamp}.csv")
        
        # Find missing leads
        logger.info("Running missing leads checker...")
        missing_leads = missing_leads_checker(
            processed_attio_data,
            cdr_fyi_purchasers,
            output_path=missing_leads_path,
            fuzzy_path=fuzzy_matches_path,
            threshold=60
        )
        
        # Write missing leads to Google Sheet
        logger.info("Writing results to Google Sheet...")
        sheet_success = write_leads_to_google_sheet(missing_leads, google_sheet_id)
        
        # Upload results to S3 if running in Lambda
        if not is_local:
            bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
            if bucket_name:
                logger.info(f"Uploading results to S3 bucket {bucket_name}...")
                upload_to_s3(missing_leads_path, bucket_name=bucket_name)
                upload_to_s3(fuzzy_matches_path, bucket_name=bucket_name)
                logger.info("Files uploaded to S3 successfully")
        
        logger.info("Missing leads check completed successfully")
        return {
            'success': True,
            'missing_leads_count': len(missing_leads),
            'google_sheet_updated': sheet_success,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Error in missing leads checker: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler to run the missing leads checker and update Google Sheets"""
    
    logger.info("Starting Missing Leads Checker Lambda")
    try:
        result = process_missing_leads(is_local=False)
        
        if result['success']:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Missing leads check completed successfully',
                    'missing_leads_count': result['missing_leads_count'],
                    'google_sheet_updated': result['google_sheet_updated'],
                    'timestamp': result['timestamp']
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': result['error']
                })
            }
        
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Failed to check missing leads: {str(e)}"
            })
        }

if __name__ == "__main__":
    # When running locally, ensure directories exist
    for directory in [RAW_DIR, PROCESSED_DIR, SALES_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Run the missing leads checker locally
    print("Running Missing Leads Checker locally...")
    result = process_missing_leads(is_local=True)
    
    if result['success']:
        print(f"Success! Found {result['missing_leads_count']} missing leads.")
        print(f"Google Sheet updated: {result['google_sheet_updated']}")
        print(f"Completed at: {result['timestamp']}")
    else:
        print(f"Failed: {result['error']}")