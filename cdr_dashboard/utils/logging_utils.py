import logging
import logging.config
import os
from pathlib import Path
import boto3
from io import StringIO
import warnings
from datetime import datetime
import time

# Add custom log levels for success and summary
logging.SUCCESS = 25  # Between INFO and WARNING
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

def success(self, message, *args, **kwargs):
    self.log(logging.SUCCESS, message, *args, **kwargs)

logging.Logger.success = success

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and symbols"""
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + "🔍 %(message)s" + reset,
        logging.INFO: grey + "ℹ️ %(message)s" + reset,
        logging.SUCCESS: green + "✅ %(message)s" + reset,
        logging.WARNING: yellow + "\u26A0\uFE0F  %(message)s" + reset,
        logging.ERROR: red + "❌ %(message)s" + reset,
        logging.CRITICAL: bold_red + "🚨 %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Add these constants at the top
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s\n' \
            'File: %(filename)s:%(lineno)d\n' \
            'Function: %(funcName)s\n' \
            '%(message)s\n'

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Define paths for the CDR dashboard project
CDR_PROJECT_ROOT = Path(__file__).parent.parent

# Define paths differently based on environment
is_lambda = os.getenv('AWS_LAMBDA_FUNCTION_NAME') is not None

if is_lambda:
    # In Lambda, use the /tmp directory which is writable
    CDR_DATA_DIR = Path('/tmp')
    CDR_LOGS_DIR = Path('/tmp/logs')
else:
    # For local execution, use project directory
    CDR_DATA_DIR = CDR_PROJECT_ROOT / 'data'
    CDR_LOGS_DIR = CDR_DATA_DIR / 'logs'
    # Create directories if they don't exist (for local runs)
    CDR_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging(name='CDRDataRetriever', level='INFO'):
    """
    Configure logging with consistent formatting
    
    Parameters
    ----------
    name : str
        Logger name (default: 'CDRDataRetriever')
    level : str
        Logging level (default: 'INFO')
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set level
    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))
    
    # Configure root logger to suppress third-party logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Suppress specific loggers
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Set up our custom logger
    logger = logging.getLogger(name)
    
    # Console handler with custom formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # File handler with detailed formatting for debugging
    file_formatter = logging.Formatter(LOG_FORMAT)
    
    # Handle both local and S3 logging
    if is_lambda:
        s3 = boto3.client('s3')
        log_buffer = StringIO()
        bucket_name = os.environ.get('CDR_DASHBOARD_BUCKET')
        
        class S3Handler(logging.StreamHandler):
            def __init__(self, buffer, s3_client, bucket, prefix):
                super().__init__(buffer)
                self.buffer = buffer
                self.s3_client = s3_client
                self.bucket = bucket
                self.prefix = prefix
            
            def flush(self):
                try:
                    if self.buffer.getvalue():
                        # Use a single consistent key for the log file
                        key = f"{self.prefix}/retrieval.log"
                        
                        # Try to get existing log content
                        try:
                            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                            existing_content = response['Body'].read().decode('utf-8')
                        except Exception:
                            existing_content = ""
                        
                        # Combine existing content with new logs
                        new_content = existing_content + self.buffer.getvalue()
                        
                        # Write back to S3
                        self.s3_client.put_object(
                            Bucket=self.bucket,
                            Key=key,
                            Body=new_content
                        )
                        
                        # Clear the buffer
                        self.buffer.truncate(0)
                        self.buffer.seek(0)
                except Exception as e:
                    print(f"Error writing logs to S3: {e}")
        
        if bucket_name:
            file_handler = S3Handler(
                buffer=log_buffer,
                s3_client=s3,
                bucket=bucket_name,
                prefix='logs'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        
    else:
        # Make sure the directory exists
        CDR_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(CDR_LOGS_DIR / 'retrieval.log')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # More detailed in files
        logger.addHandler(file_handler)

    logger.propagate = False
    
    return logger

def log_with_timestamp(logger, level, message):
    """Log with a timestamp prepended"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if level == 'info':
        logger.info(f"[{timestamp}] {message}")
    elif level == 'success':
        logger.success(f"[{timestamp}] {message}")
    elif level == 'warning':
        logger.warning(f"[{timestamp}] {message}")
    elif level == 'error':
        logger.error(f"[{timestamp}] {message}")
    elif level == 'debug':
        logger.debug(f"[{timestamp}] {message}")