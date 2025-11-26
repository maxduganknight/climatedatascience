import os
import json
import boto3
import re
from botocore.exceptions import ClientError
from datetime import datetime, timedelta

def handler(event, context):
    print('Event:', json.dumps(event))
    
    # Get allowed origins - include localhost for testing
    allowed_origins = [
        'https://www.deepskyclimate.com',
        'https://deepskyclimate.com',
        'http://localhost:8000',
        'https://deepsky.design.webflow.com',
        'https://www.deepsky.design.webflow.com',
        'https://deepsky.webflow.io'
    ]
    origin = event.get('headers', {}).get('origin', 'https://www.deepskyclimate.com')
    
    # Always return CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': origin if origin in allowed_origins else allowed_origins[0],
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type,Origin'
    }
    
    # Handle preflight OPTIONS request
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        s3 = boto3.client('s3')
        bucket_name = os.environ.get('DASHBOARD_METRICS_BUCKET')
        print(f'Using bucket: {bucket_name}')
        
        requested_key = event.get('queryStringParameters', {}).get('key') if event.get('queryStringParameters') else None
        print(f'Requested key: {requested_key}')
        
        if requested_key:
            try:
                # Fix: Check if the key already contains 'processed/' to avoid duplication
                if requested_key.startswith('viz/'):
                    s3_key = requested_key  # Keep viz/ paths as is
                    content_type = 'application/javascript' if requested_key.endswith('.js') else 'text/csv'
                elif requested_key.startswith('processed/'):
                    s3_key = requested_key  # Already has processed/ prefix
                    content_type = 'application/json' if requested_key.endswith('.json') else 'text/csv'
                else:
                    s3_key = f'processed/{requested_key}'  # Add processed/ prefix for other files
                    content_type = 'application/json' if requested_key.endswith('.json') else 'text/csv'
                
                print(f'Processing request for key: {s3_key} with content type: {content_type}')  # Debug log
                
                if '*' in requested_key:
                    # Handle wildcard pattern
                    prefix = s3_key.split('*')[0]
                    print(f'Looking for files with prefix: {prefix}')
                    
                    response = s3.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=prefix
                    )
                    
                    # Find the most recent file matching the pattern
                    latest_file = None
                    latest_date = None
                    
                    for obj in response.get('Contents', []):
                        key = obj['Key']
                        # Extract date from filename using regex (e.g., 20240211)
                        match = re.search(r'(\d{8})', key)
                        if match:
                            date_str = match.group(1)
                            try:
                                file_date = datetime.strptime(date_str, '%Y%m%d')
                                if latest_date is None or file_date > latest_date:
                                    latest_date = file_date
                                    latest_file = key
                            except ValueError:
                                continue
                    
                    if latest_file is None:
                        return {
                            'statusCode': 404,
                            'headers': headers,
                            'body': json.dumps({'error': f'No matching files found for pattern: {requested_key}'})
                        }
                    
                    s3_key = latest_file
                    print(f'Found latest matching file: {s3_key}')
                
                # Generate presigned URL
                params = {
                    'Bucket': bucket_name,
                    'Key': s3_key,
                    'ResponseContentType': content_type,
                    'ResponseContentDisposition': f'inline; filename="{requested_key}"'
                }
                
                signed_url = s3.generate_presigned_url(
                    'get_object',
                    Params=params,
                    ExpiresIn=300
                )
                print(f'Generated presigned URL for {s3_key}')
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'url': signed_url})
                }
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == '404' or error_code == 'NoSuchKey':
                    print(f'File not found: {s3_key}')
                    return {
                        'statusCode': 404,
                        'headers': headers,
                        'body': json.dumps({'error': f'File not found: {requested_key}'})
                    }
                raise
        else:
            # List available files
            try:
                response = s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix='processed/'
                )
                
                files = [
                    obj['Key'].replace('processed/', '')
                    for obj in response.get('Contents', [])
                    if obj['Key'] != 'processed/'
                ]
                
                print(f'Found {len(files)} files')
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'files': files})
                }
            except ClientError as e:
                print(f'Error listing files: {str(e)}')
                raise
            
    except Exception as e:
        print('Error:', str(e))
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Failed to process request',
                'details': str(e)
            })
        }