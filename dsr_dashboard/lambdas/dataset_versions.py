import os
import sys
import json
import re
import boto3
from botocore.exceptions import ClientError
from datetime import datetime

# Add parent directory to Python path 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_dataset_versions():
    """Get the latest version (date) for each dataset in S3"""
    try:
        bucket_name = os.environ.get('DASHBOARD_METRICS_BUCKET')
        if not bucket_name:
            print("ERROR: DASHBOARD_METRICS_BUCKET environment variable not set")
            return {}
            
        s3_client = boto3.client('s3')
        
        print(f"Listing objects in bucket: {bucket_name} with prefix: processed/")
        
        # List all processed files in the bucket
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='processed/'
        )
        
        # Group files by dataset and find the latest version
        dataset_versions = {}
        
        if 'Contents' in response:
            for item in response['Contents']:
                key = item['Key']
                if key.endswith('.csv'):
                    # Extract dataset name and date using regex
                    match = re.search(r'processed/(.+?)_(\d{8})\.csv', key)
                    if match:
                        dataset_name = match.group(1)
                        version_date = match.group(2)
                        
                        # Update if this is a newer version
                        if dataset_name not in dataset_versions or version_date > dataset_versions[dataset_name]:
                            dataset_versions[dataset_name] = version_date
        
        print(f"Found {len(dataset_versions)} datasets with versions")
        return dataset_versions
    except Exception as e:
        print(f"ERROR in get_dataset_versions: {str(e)}")
        # Return empty dict instead of failing
        return {}

def handler(event, context):
    try:
        print(f"Received event: {json.dumps(event)}")
        
        # Get allowed origins - include localhost for testing
        allowed_origins = [
            'https://www.deepskyclimate.com',
            'https://deepskyclimate.com',
            'http://localhost:8000',
            'https://deepsky.design.webflow.com',
            'https://www.deepsky.design.webflow.com',
            'https://deepsky.webflow.io'
        ]
        
        # Default headers to ensure we always return valid CORS headers
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': 'https://www.deepskyclimate.com',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type,Origin'
        }
        
        # Get origin from request if available
        origin = event.get('headers', {}).get('origin')
        if origin in allowed_origins:
            headers['Access-Control-Allow-Origin'] = origin
        
        # Handle preflight OPTIONS request
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': ''
            }
        
        # Get dataset versions from S3
        dataset_versions = get_dataset_versions()
        
        # Format versions as expected by the client
        versions = {}
        for dataset_name, version_date in dataset_versions.items():
            versions[f"{dataset_name}_*.csv"] = version_date
        
        # Add last updated timestamp
        response_body = {
            "versions": versions,
            "server_time": datetime.utcnow().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        
        # Ensure we have headers defined even in case of error
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': 'https://www.deepskyclimate.com',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type,Origin'
        }
        
        # Get origin from request if available
        try:
            origin = event.get('headers', {}).get('origin')
            if origin in allowed_origins:
                headers['Access-Control-Allow-Origin'] = origin
        except:
            pass
            
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Failed to retrieve dataset versions',
                'details': str(e)
            })
        }

if __name__ == "__main__":
    # For local testing
    result = handler({}, None)
    print(json.dumps(json.loads(result['body']), indent=2))