import logging
import logging.config
import yaml
import os
from pathlib import Path
import boto3
from io import StringIO
import warnings
from .paths import PROJECT_ROOT, DATA_DIR, LOGS_DIR
from datetime import datetime
import time
import functools

# Add custom log levels for success and summary
logging.SUCCESS = 25  # Between INFO and WARNING
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

def success(self, message, *args, **kwargs):
    self.log(logging.SUCCESS, message, *args, **kwargs)

logging.Logger.success = success

# Add dataset-specific logging methods
def log_dataset_operation(self, dataset_name, operation, level='info', **kwargs):
    """
    Log operations with clear dataset context.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset being processed
    operation : str
        Description of the operation being performed
    level : str
        Log level ('info', 'warning', 'error', 'success')
    **kwargs : 
        Additional information to include in the log message
    """
    # Create a dataset identifier that stands out
    dataset_id = f"[{dataset_name}]"
    
    # Format any additional information
    additional_info = ""
    if kwargs:
        additional_info = " - " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
    
    # Create the message with dataset context
    message = f"{dataset_id} {operation}{additional_info}"
    
    # Log at the appropriate level
    if level.lower() == 'warning':
        self.warning(message)
    elif level.lower() == 'error':
        self.error(message)
    elif level.lower() == 'success':
        self.success(message)
    else:  # default to info
        self.info(message)

logging.Logger.log_dataset_operation = log_dataset_operation

# Add method to create section headers in logs
def log_section(self, title, dataset_name=None, width=60):
    """
    Create a visual section header in the logs.
    
    Parameters:
    -----------
    title : str
        Title of the section
    dataset_name : str, optional
        Name of the dataset, if applicable
    width : int
        Width of the section header
    """
    separator = "=" * width
    
    if dataset_name:
        header = f"{dataset_name} - {title}"
    else:
        header = title
        
    # Center the header within the separator
    padded_header = f" {header} ".center(width, "=")
    
    self.info(separator)
    self.info(padded_header)
    self.info(separator)

logging.Logger.log_section = log_section

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

# Add a utility to create a dataset logger (a wrapper around setup_logging)
def setup_dataset_logger(dataset_name, level='INFO'):
    """
    Set up a logger specifically for a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset for which the logger is being created
    level : str
        Logging level (default: 'INFO')
        
    Returns:
    --------
    logger : logging.Logger
        Configured logger with dataset context methods
    """
    logger = setup_logging(name=f"Dataset.{dataset_name}", level=level)
    
    # Create a special method that properly handles dataset operations
    def dataset_op(operation, level='info', **kwargs):
        """Log dataset operations with proper context"""
        logger.log_dataset_operation(dataset_name, operation, level, **kwargs)
    
    # Add the method directly instead of using functools.partial
    logger.dataset_op = dataset_op
    
    # Create a special method for sections that automatically includes dataset name
    def section_with_dataset(title, width=60):
        """Create section headers with dataset context"""
        logger.log_section(title, dataset_name=dataset_name, width=width)
    
    # Replace the section method
    logger.section = section_with_dataset
    
    return logger

def setup_logging(name='DataRetriever', level='INFO'):
    """
    Configure logging with consistent formatting
    
    Parameters
    ----------
    name : str
        Logger name (default: 'DataRetriever')
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
    logging.getLogger('datapi').setLevel(logging.ERROR)
    logging.getLogger('cdsapi').setLevel(logging.ERROR)
    
    # Console handler with custom formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # File handler with detailed formatting for debugging
    file_formatter = logging.Formatter(LOG_FORMAT)
    
    # Handle both local and S3 logging
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        s3 = boto3.client('s3')
        log_buffer = StringIO()
        
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
                        except self.s3_client.exceptions.NoSuchKey:
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
        
        file_handler = S3Handler(
            buffer=log_buffer,
            s3_client=s3,
            bucket=str(DATA_DIR),
            prefix='logs'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        
    else:
        file_handler = logging.FileHandler(LOGS_DIR / 'retrieval.log')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # More detailed in files
        logger.addHandler(file_handler)

    logger.propagate = False
    
    # Configure warnings more aggressively
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', module='cdsapi')
    warnings.filterwarnings('ignore', module='datapi')
    
    return logger