import os
import sys
import json
import glob
import re
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add dashboard directory to Python path
dashboard_dir = Path(__file__).parent.parent
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def do_GET(self):
        # Parse URL and query parameters
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Handle dataset-versions endpoint for local testing
        if parsed_url.path == '/dataset-versions':
            base_dir = os.path.join(dashboard_dir, 'data', 'processed')
            
            # Build a dictionary of the latest version for each dataset
            versions = {}
            file_pattern = re.compile(r'([a-z0-9_]+)_(\d{8})\.csv')
            
            for file in os.listdir(base_dir):
                match = file_pattern.match(file)
                if match:
                    base_name = match.group(1)
                    version = match.group(2)
                    
                    # Pattern with wildcard (as used in the frontend)
                    pattern_key = f"{base_name}_*.csv"
                    
                    # Keep the latest version
                    if pattern_key not in versions or version > versions[pattern_key]:
                        versions[pattern_key] = version
            
            # Send JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'versions': versions}).encode())
            return
            
        # If there's a pattern parameter, handle file pattern matching
        elif parsed_url.path.startswith('/data/processed/') and 'pattern' in query_params:
            pattern = query_params['pattern'][0]
            base_dir = os.path.join(dashboard_dir, 'data', 'processed')
            
            # Find matching files
            search_pattern = os.path.join(base_dir, f"{pattern}*.csv")
            matching_files = glob.glob(search_pattern)
            filenames = [os.path.basename(f) for f in matching_files]
            filenames.sort()  # Sort to get most recent last
            
            # Send JSON response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(filenames).encode())
            return
            
        # For all other requests, use default handler
        return super().do_GET()

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print('Starting local server at http://localhost:8000')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()