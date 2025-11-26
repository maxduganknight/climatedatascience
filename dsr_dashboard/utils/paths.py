import os
import sys
from pathlib import Path

def get_project_root() -> Path:
    """Get the root directory of the dashboard project"""
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        # In Lambda environment
        return Path('/var/task')
    else:
        # In local environment, navigate up from this file to dashboard root
        return Path(__file__).parent.parent

class S3Path:
    """Class to handle S3 paths similar to Path objects"""
    def __init__(self, bucket: str):
        self.bucket = bucket

    def __truediv__(self, other):
        """Handle / operator to join paths"""
        return S3Path(f"{self.bucket}/{other}")

    def __str__(self):
        return self.bucket

def get_data_dir():
    """Get the main data directory"""
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        bucket = os.environ.get('DASHBOARD_METRICS_BUCKET')
        return S3Path(bucket)
    
    else:
        return get_project_root() / 'data'

# Add current directory to Python path if not already there
if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    dashboard_dir = get_project_root()
    if str(dashboard_dir) not in sys.path:
        sys.path.insert(0, str(dashboard_dir))

# Define commonly used paths
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_dir()
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
CONFIG_PATH = PROJECT_ROOT / 'config' / 'dataset_dir.json'
LOGS_DIR = DATA_DIR / 'logs'

# Define S3 paths for Lambda environment
if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    S3_BUCKET = os.environ['DASHBOARD_METRICS_BUCKET']
    S3_RAW_PREFIX = 'raw'
    S3_PROCESSED_PREFIX = 'processed'
    S3_LOGS_PREFIX = 'logs'
else:
    S3_BUCKET = None
    S3_RAW_PREFIX = None
    S3_PROCESSED_PREFIX = None
    S3_LOGS_PREFIX = None

# Create directories if they don't exist (only in local environment)
if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    for directory in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)