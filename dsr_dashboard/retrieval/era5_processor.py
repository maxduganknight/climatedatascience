import sys
from pathlib import Path
# Add dashboard directory to Python path
dashboard_dir = Path(__file__).parent.parent
sys.path.append(str(dashboard_dir))

import pandas as pd
import numpy as np
import xarray as xr
import json
import datetime
import logging
import boto3
import glob
import os
from utils.logging_utils import setup_logging
from utils.era5_utils import create_time_dataframe, calculate_baseline_values
from utils.retrieval_utils import save_dataset, load_config
from utils.paths import DATA_DIR, RAW_DIR, PROCESSED_DIR

setup_logging()
logger = logging.getLogger(__name__)

def process_era5_temp_data(ds, var_name, time_step, data_dir=None, preindustrial_baseline=False, is_local=True):
    """
    Convert ERA5 raw temperature data in .nc files to month-day anomaly data in csv files for plotting.  
    Air temperature data is adjusted to pre-industrial baseline in order to show us moving past 1.5C.
    Sea surface temperature data is not adjusted and uses 1991-2020 baseline values because pre-industrial adjustment data is not available.
    T2m adjustment values come from: https://climate.copernicus.eu/tracking-breaches-150c-global-warming-threshold. 
    """
    # Print diagnostic info about input dataset
    logger.info(f"Input dataset dimensions: {ds.dims}")
    logger.info(f"Valid time range: {pd.to_datetime(ds.valid_time.values[0])} to {pd.to_datetime(ds.valid_time.values[-1])}")
    logger.info(f"Number of time points: {len(ds.valid_time)}")
    
    df = create_time_dataframe(ds[var_name].values, ds['valid_time'].values)
    
    # Print basic stats about the dataframe
    logger.info(f"DataFrame shape after creation: {df.shape}")
    logger.info(f"Unique years: {sorted(df['year'].unique())}")
    logger.info(f"Unique month-days: {len(df['month_day'].unique())}")
    
    # Drop February 29th data
    if time_step == 'month_day':
        df = df[df['month_day'] != '02-29']
    
    # Handle preindustrial baseline adjustments
    if preindustrial_baseline:
        adjustment_var = f'{var_name}_adjustment'
        if time_step == 'month_day':
            # Load monthly adjustments
            if is_local:
                preindustrial_baseline_adjustment = pd.read_csv(data_dir / 'raw/pre_industrial_adjustments.csv')
            else:
                # For S3 environment
                s3 = boto3.client('s3')
                bucket = str(DATA_DIR)
                try:
                    response = s3.get_object(
                        Bucket=bucket,
                        Key='raw/pre_industrial_adjustments.csv'
                    )
                    preindustrial_baseline_adjustment = pd.read_csv(response['Body'])
                except Exception as e:
                    logger.error(f"Failed to read preindustrial baseline adjustment file from S3: {e}")
                    raise
            
            df = df.merge(preindustrial_baseline_adjustment[[adjustment_var, 'month']], on='month', how='left')
        elif time_step == 'year':
            # Use single adjustment value for t2m annual data
            # MDK if I ever want to do annual SST I will need to change this
            df[adjustment_var] = 0.88

    # Calculate baseline and anomalies
    if time_step == 'year':
        # For annual data, first calculate yearly means while preserving adjustment
        if preindustrial_baseline:
            df = df.groupby('year').agg({
                'value': 'mean',
                adjustment_var: 'first'  # Keep the adjustment value
            }).reset_index()
        else:
            df = df.groupby('year')['value'].mean().reset_index()
        
        baseline_value = df[(df['year'] >= 1991) & (df['year'] <= 2020)]['value'].mean()
        df['value_baseline'] = baseline_value
    elif var_name == 't2m':
        # For monthly data, calculate baseline for each month-day
        baseline_values = calculate_baseline_values(df, 1991, 2020)
        df = df.merge(baseline_values.reset_index(), on='month_day', how='left', suffixes=('', '_baseline'))
    elif var_name == 'sst':
        # For SST, use a different baseline calculation
        baseline_values = calculate_baseline_values(df, 2011, 2020)
        df = df.merge(baseline_values.reset_index(), on='month_day', how='left', suffixes=('', '_baseline'))

    # Add detailed debug info
    logger.info(f"DataFrame shape before anomaly calculation: {df.shape}")
    
    # Calculate anomalies with or without adjustment
    if preindustrial_baseline:
        df['anom'] = df['value'] - df['value_baseline'] + df[adjustment_var]
    else:
        df['anom'] = df['value'] - df['value_baseline']
    
    # Debug: print counts of month_day-year combinations
    if time_step == 'month_day':
        # Check for duplicate month_day/year combinations
        counts = df.groupby(['month_day', 'year']).size().reset_index(name='count')
        duplicates = counts[counts['count'] > 1]
        
        if len(duplicates) > 0:
            logger.error(f"Found {len(duplicates)} duplicate month_day/year combinations:")
            for _, row in duplicates.iterrows():
                logger.error(f"  month_day={row['month_day']}, year={row['year']}, count={row['count']}")
                
                # Print the specific records causing the issue
                problem_records = df[(df['month_day'] == row['month_day']) & (df['year'] == row['year'])]
                logger.error(f"  Problem records:\n{problem_records[['month_day', 'year', 'time', 'value', 'anom']].to_string()}")
        else:
            logger.info("No duplicate month_day/year combinations found")
    
    # Format output based on time_step
    if time_step == 'year':
        # For annual data, return simple year-value format
        output_df = df[['year', 'anom']].set_index('year')
    else:
        # Fix for month_day data - ensure exactly one value per month_day/year combination
        logger.info(f"Before pivot, df shape: {df.shape}")
        
        # Use first() aggregation to get one value per month_day/year combination
        logger.info("Aggregating to ensure one value per month_day/year combination...")
        df_agg = df.groupby(['month_day', 'year'])['anom'].mean().reset_index()
        
        logger.info(f"After aggregation, shape: {df_agg.shape}")
        logger.info(f"Unique month_day values: {len(df_agg['month_day'].unique())}")
        logger.info(f"Unique year values: {len(df_agg['year'].unique())}")
        
        # For monthly data, pivot to wide format with years as columns
        logger.info("Performing pivot operation...")
        output_df = df_agg.pivot(index='month_day', columns='year', values='anom')
        logger.info(f"Pivot successful, output shape: {output_df.shape}")
    
    return output_df.sort_index()

class ERA5Processor:
    def __init__(self, is_local=True):
        self.is_local = is_local
        self.logger = setup_logging()
        self.dataset_dir = load_config()
    
    def _process_era5_data(self, ds: xr.Dataset, dataset_name: str) -> pd.DataFrame:
        """Process ERA5 dataset into anomaly data"""
        config = self.dataset_dir[dataset_name]
        var_name = config['era5_nc_variable']
        time_step = config['x_axis_unit']
        preindustrial_baseline = config.get('preindustrial_baseline', False)
        
        return process_era5_temp_data(
            ds=ds,
            var_name=var_name,
            time_step=time_step,
            data_dir=DATA_DIR,
            preindustrial_baseline=preindustrial_baseline,
            is_local=self.is_local
        )

    def _save_processed_data(self, data: pd.DataFrame, dataset_name: str):
        """Save processed data to file"""
        save_dataset(data, dataset_name, is_local=self.is_local)

    def process_dataset(self, dataset_name: str):
        """Process a specific ERA5 dataset"""
        logger.info(f"Processing {dataset_name}")
        
        try:
            # Get the most recent file for this dataset
            if self.is_local:
                pattern = RAW_DIR / f'{dataset_name}_*.nc'
                files = glob.glob(str(pattern))
                if not files:
                    raise FileNotFoundError(f"No files found matching {pattern}")
                latest_file = max(files)  # Get most recent file
                
                # Process the file directly
                with xr.open_dataset(latest_file) as ds:
                    data = self._process_era5_data(ds, dataset_name)
                    
            else:
                # For S3, use boto3 to get and process the file
                s3 = boto3.client('s3')
                bucket = str(DATA_DIR)
                prefix = f'raw/{dataset_name}_'
                
                # List objects in the bucket
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
                )
                
                if 'Contents' not in response:
                    raise FileNotFoundError(f"No files found in S3 with prefix {prefix}")
                    
                # Get the most recent file
                files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.nc')]
                if not files:
                    raise FileNotFoundError(f"No .nc files found in S3 with prefix {prefix}")
                    
                latest_file = max(files)
                logger.info(f"Processing file: {latest_file}")
                
                # Download to temporary file
                temp_file = f'/tmp/{os.path.basename(latest_file)}'
                try:
                    s3.download_file(bucket, latest_file, temp_file)
                    
                    # Process the temporary file
                    with xr.open_dataset(temp_file) as ds:
                        data = self._process_era5_data(ds, dataset_name)
                        
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.info(f"Removed temporary file: {temp_file}")
            
            # Save the processed data
            self._save_processed_data(data, dataset_name)
            logger.success(f"Successfully processed {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {str(e)}")
            raise
    
    def process_all_datasets(self):
        """Process all ERA5 datasets"""
        era5_datasets = [name for name in self.dataset_dir.keys() if name.startswith('era5_')]

        for dataset in era5_datasets:
            try:
                self.process_dataset(dataset)
            except Exception as e:
                self.logger.error(f"Failed to process {dataset}: {str(e)}")

def main():
    processor = ERA5Processor(is_local=True)
    processor.process_all_datasets()

if __name__ == '__main__':
    main()
