import os
import json
import boto3
from urllib.parse import unquote
import logging
import re
from botocore.exceptions import ClientError
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """Lambda handler for generating presigned URLs for S3 objects"""
    logger.info("CDR DASHBOARD PRESIGNED URL LAMBDA INVOKED")
    logger.info(f'Event: {json.dumps(event)}')
    
    # Get allowed origins - include localhost for testing and Webflow domains
    allowed_origins = [
        'https://www.deepskyclimate.com',
        'https://fr.deepskyclimate.com',
        'https://deepskyclimate.com',
        'http://localhost:8000',
        'https://deepsky.design.webflow.com',
        'https://deepsky.webflow.io',  # Without trailing slash
        'https://deepsky.webflow.io/',  # With trailing slash
        'https://www.deepsky.design.webflow.com',
        'https://www.webflow.com',
        'null'  # Add this for local file testing
    ]
    
    # Get origin from request headers with better fallback logic
    headers_dict = event.get('headers', {}) or {}
    origin = headers_dict.get('origin') or headers_dict.get('Origin') or headers_dict.get('referer') or headers_dict.get('Referer')

    # Add more normalization to handle trailing slashes
    if origin:
        # Strip trailing slash for comparison
        normalized_origin = origin.lower().rstrip('/')
        normalized_allowed_origins = [o.lower().rstrip('/') for o in allowed_origins]
    
    # Debug the origin
    logger.info(f"Received request with Origin: {origin}")
    
    # Make the origin comparison case-insensitive and handle null origin
    if origin == 'null' or not origin:
        # For local file testing or when origin is missing
        matching_origin = '*'
    else:
        found_match = False
        for o, no in zip(allowed_origins, normalized_allowed_origins):
            if no == normalized_origin:
                matching_origin = o
                found_match = True
                break
        
        if not found_match:
            # If no match, default to the first allowed origin
            matching_origin = allowed_origins[0]
    
    logger.info(f"Using Access-Control-Allow-Origin: {matching_origin}")
    
    # Set CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': matching_origin,
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type,Origin,X-Requested-With'
    }
    
    # Handle preflight OPTIONS request - THIS IS CRUCIAL
    if event.get('httpMethod') == 'OPTIONS':
        logger.info('Handling OPTIONS preflight request')
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # Get parameters from the request
        query_params = event.get('queryStringParameters', {}) or {}
        requested_key = query_params.get('key')
        
        # Default bucket from environment or query parameter
        default_bucket = os.environ.get('CDR_DASHBOARD_BUCKET')
        bucket = query_params.get('bucket', default_bucket)
        
        logger.info(f'Using bucket: {bucket}, requested key: {requested_key}')
        
        if not requested_key:
            logger.error('No key specified in request')
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Missing required parameter: key'})
            }
        
        # THIS IS KEY - Special handling for viz/ paths vs processed/ paths
        if requested_key.startswith('viz/'):
            s3_key = requested_key  # Keep viz/ paths as is
            content_type = 'application/javascript' if requested_key.endswith('.js') else 'text/css'
        elif requested_key.startswith('processed/'):
            s3_key = requested_key  # Already has processed/ prefix
            content_type = 'application/json' if requested_key.endswith('.json') else 'text/csv'
        else:
            s3_key = f'processed/{requested_key}'  # Add processed/ prefix for data files
            content_type = 'text/csv'
        
        # Handle file patterns with wildcards
        if '*' in requested_key:
            logger.info(f'Handling wildcard pattern: {requested_key}')
            s3_key = find_latest_matching_file(bucket, s3_key)
            
            if not s3_key:
                return {
                    'statusCode': 404,
                    'headers': headers,
                    'body': json.dumps({'error': f'No matching files found for pattern: {requested_key}'})
                }
                
        # Generate presigned URL
        presigned_url = generate_presigned_url(bucket, s3_key, content_type)
        
        if not presigned_url:
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({'error': f'Failed to generate URL for: {s3_key}'})
            }
            
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'url': presigned_url})
        }
        
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Failed to process request',
                'details': str(e)
            })
        }

def generate_presigned_url(bucket, key, content_type=None, expiration=300):
    """Generate a presigned URL for an S3 object"""
    try:
        # Debug info
        logger.info(f"Generating presigned URL for bucket={bucket}, key={key}, content_type={content_type}")
        
        # Generate the presigned URL
        s3_client = boto3.client('s3')
        
        params = {
            'Bucket': bucket,
            'Key': key
        }
        
        # Explicit content type handling is important for script loading
        if content_type:
            params['ResponseContentType'] = content_type
            params['ResponseContentDisposition'] = f'inline; filename="{os.path.basename(key)}"'
        
        # Add cache-busting parameters
        # For JavaScript files, add a cache control header to prevent caching
        if key.endswith('.js'):
            params['ResponseCacheControl'] = 'no-store, no-cache, must-revalidate, max-age=0'
            
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=expiration
        )
        
        logger.info(f"Generated presigned URL (truncated): {presigned_url[:100]}...")
        return presigned_url
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404' or error_code == 'NoSuchKey':
            logger.error(f'File not found: {key}')
            return None
        raise
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return None

def find_latest_matching_file(bucket, pattern):
    """Find the latest file matching a pattern with wildcard"""
    try:
        s3_client = boto3.client('s3')
        
        # Split at wildcard like in climate dashboard
        prefix = pattern.split('*')[0]
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        # Find the most recent file matching the pattern (similar to climate dashboard)
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
        
        return latest_file
        
    except Exception as e:
        logger.error(f"Error finding latest matching file: {str(e)}")
        return None