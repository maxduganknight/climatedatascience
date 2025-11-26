import os
import sys
import json
import glob
import re
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add cdr_dashboard directory to Python path
cdr_dashboard_dir = Path(__file__).parent.parent
if str(cdr_dashboard_dir) not in sys.path:
    sys.path.append(str(cdr_dashboard_dir))

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def do_GET(self):
        # Parse URL and query parameters
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Change directory to serve files from the root of the cdr_dashboard project
        os.chdir(cdr_dashboard_dir)
        
        # For all requests, use default handler with CORS support
        return super().do_GET()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f'Starting local CDR dashboard server at http://localhost:{port}')
    print(f'Serving files from: {cdr_dashboard_dir}')
    print(f'Access test page at: http://localhost:{port}/viz/test_local.html')
    httpd.serve_forever()

if __name__ == '__main__':
    # Allow specifying a different port via command line argument
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Using default port 8000.")
    
    run_server(port)
