import os
from pathlib import Path

# Check if running in Lambda
IS_LAMBDA = os.getenv('AWS_LAMBDA_FUNCTION_NAME') is not None

# Define project root and environment-specific paths
PROJECT_ROOT = Path(__file__).parent.parent

if IS_LAMBDA:
    # In Lambda, use /tmp which is writable
    DATA_DIR = Path('/tmp')
    RAW_DIR = Path('/tmp/raw')
    PROCESSED_DIR = Path('/tmp/processed')
    LOGS_DIR = Path('/tmp/logs')
    SALES_DIR = Path('/tmp/sales')
else:
    # For local execution
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DIR = DATA_DIR / 'raw'
    PROCESSED_DIR = DATA_DIR / 'processed'
    LOGS_DIR = DATA_DIR / 'logs'
    SALES_DIR = PROJECT_ROOT / 'sales/data'
    
    # Create directories for local development
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)