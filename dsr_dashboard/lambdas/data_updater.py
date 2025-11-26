import sys
from pathlib import Path
# Add dashboard directory to Python path
dashboard_dir = Path(__file__).parent.parent
if str(dashboard_dir) not in sys.path:
    sys.path.append(str(dashboard_dir))

import os
from datetime import datetime
from utils.logging_utils import setup_logging
from utils.retrieval_utils import load_config, needs_update, get_aws_secret
from retrieval.aviso_slr import process_aviso_slr
from retrieval.noaa_billion import process_noaa_billion
from retrieval.co2_ppm import process_co2_ppm
# from retrieval.ocean_ph import process_ocean_ph
from retrieval.era5_retriever import DatasetRetriever
from retrieval.era5_processor import ERA5Processor
from retrieval.home_insurance_premium import process_home_insurance_premium
from retrieval.arctic_sea_ice import process_arctic_sea_ice  # Import the new retrieval script
from utils.paths import RAW_DIR, PROCESSED_DIR, LOGS_DIR
import logging

logger = setup_logging()

def update_datasets():
    """Update all datasets"""
    is_local = not bool(os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
    datasets_updated = []
    
    try:
        logger.info("\n📊 Starting Dataset Updates 📊")
        logger.info("=" * 50)
        
        # Process individual datasets
        for dataset, process_func in [
            ('aviso_slr', process_aviso_slr),
            ('noaa_billion', process_noaa_billion),
            ('co2_ppm', process_co2_ppm),
            ('home_insurance_premium', process_home_insurance_premium),
            ('arctic_sea_ice', process_arctic_sea_ice)  # Add the new dataset
        ]:
            try:
                config = load_config()
                if needs_update(dataset, config, is_local):
                    logger.info(f"Updating {dataset}...")
                    output_path = process_func(config, is_local)
                    datasets_updated.append(dataset)
                    logger.success(f"Updated {dataset} -> {output_path}\n")
                else:
                    logger.success(f"{dataset} is up to date\n")
            except Exception as e:
                logger.error(f"Failed to update {dataset}: {str(e)}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Detailed error:")
        
        # Process ERA5 datasets
        try:
            retriever = DatasetRetriever(is_local=is_local)
            processor = ERA5Processor(is_local=is_local)
            
            era5_datasets = [name for name in retriever.dataset_dir.keys() 
                           if name.startswith('era5_')]
            
            for dataset in era5_datasets:
                if needs_update(dataset, config, is_local):
                    logger.info(f"Updating {dataset}...")
                    retriever.update_era5_dataset(dataset)
                    processor.process_dataset(dataset)
                    datasets_updated.append(dataset)
                    logger.success(f"Updated {dataset}\n")
                else:
                    logger.success(f"{dataset} is up to date\n")
                    
        except Exception as e:
            logger.error(f"Failed to process ERA5 datasets: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("Detailed error:")
        
        # Summary
        logger.info("\n📋 Update Summary")
        logger.info("=" * 50)
        if datasets_updated:
            logger.success(f"Updated {len(datasets_updated)} datasets: {', '.join(datasets_updated)}")
        else:
            logger.success("All datasets are up to date")
        
        logger.info("\n✨ Update Complete ✨")
        
        return {
            'statusCode': 200,
            'updated_datasets': datasets_updated
        }
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Detailed error:")
        return {
            'statusCode': 500,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler"""
    logger = logging.getLogger('DataRetriever')
    
    # Log Lambda context information
    logger.info(f"Lambda function: {context.function_name}")
    logger.info(f"Request ID: {context.aws_request_id}")
    logger.info(f"Memory limit: {context.memory_limit_in_mb}MB")
    logger.info(f"Time remaining: {context.get_remaining_time_in_millis()}ms")
    
    try:
        result = update_datasets()
        logger.info(f"Handler completed successfully: {result}")
        return result
    except Exception as e:
        logger.error("Lambda handler failed", exc_info=True)
        raise

def ensure_local_directories():
    """Ensure required local directories exist"""
    if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        for path in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        logger.info("Local directories verified")

if __name__ == "__main__":
    ensure_local_directories()
    result = update_datasets()