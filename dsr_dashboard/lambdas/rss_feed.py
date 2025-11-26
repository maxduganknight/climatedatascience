import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from dateutil import parser
import traceback

def handler(event, context):
    # Configure CORS headers once
    allowed_origins = [
        'https://www.deepskyclimate.com',
        'https://deepskyclimate.com',
        'https://deepsky.design.webflow.com',
        'https://www.deepsky.design.webflow.com',
        'https://deepsky.webflow.io'
    ]
    origin = event.get('headers', {}).get('origin', 'https://www.deepskyclimate.com')
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': origin if origin in allowed_origins else allowed_origins[0],
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type,Origin'
    }
    
    # Handle preflight OPTIONS request
    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}
    
    try:
        # Extract RSS URL parameter
        query_params = event.get('queryStringParameters', {}) or {}
        rss_url = query_params.get('url')
        
        if not rss_url:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Missing URL parameter'})
            }
        
        # Fetch RSS feed with robust error handling
        try:
            response = requests.get(
                rss_url, 
                headers={'User-Agent': 'Mozilla/5.0 (DeepSkyClimate/1.0)'},
                timeout=5
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return {
                'statusCode': 502,
                'headers': headers,
                'body': json.dumps({'error': f'Failed to fetch RSS feed: {str(e)}'})
            }
        
        # Parse XML and extract items
        try:
            root = ET.fromstring(response.text)
            items = []
            
            for item in root.findall('.//item'):
                # Extract title and source
                full_title = item.find('title').text if item.find('title') is not None else 'No title'
                title_parts = full_title.split(' - ')
                
                if len(title_parts) > 1:
                    title = title_parts[0].strip()
                    source = title_parts[-1].strip()
                else:
                    title = full_title
                    source = item.find('source').text if item.find('source') is not None else 'Unknown source'
                
                # Parse date
                date_str = item.find('pubDate').text if item.find('pubDate') is not None else None
                formatted_date = 'Unknown date'
                if date_str:
                    try:
                        date_obj = parser.parse(date_str)
                        formatted_date = date_obj.strftime('%b %d, %Y')
                    except Exception:
                        pass
                
                items.append({
                    'title': title,
                    'source': source,
                    'date': formatted_date
                })
            
            # Sort items prioritizing notable sources and return
            priority_sources = ['The New York Times', 'Financial Times', 'CNN', 'BBC', 'Washington Post']
            sorted_items = sorted(
                items, 
                key=lambda x: (
                    any(ps.lower() in x.get('source', '').lower() for ps in priority_sources),
                    x.get('source') == 'The New York Times'  # Extra weight to NYT
                ),
                reverse=True
            )
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'items': sorted_items[:10],
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
            
        except ET.ParseError as e:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': f'Failed to parse XML: {str(e)}'})
            }
            
    except Exception as e:
        # Catch-all error handler
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e),
                'trace': traceback.format_exc()
            })
        }