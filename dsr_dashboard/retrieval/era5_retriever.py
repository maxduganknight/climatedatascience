import sys
from pathlib import Path
dashboard_dir = Path(__file__).parent.parent
sys.path.append(str(dashboard_dir))

import pandas as pd
import numpy as np
import xarray as xr
import cdsapi 
import json
import glob
import os
import datetime
import warnings
from dask.diagnostics import ProgressBar
import logging
import boto3

from utils.logging_utils import setup_logging
from utils.era5_utils import process_netcdf_data
from utils.retrieval_utils import load_config, needs_update, cleanup_old_files, get_aws_secret
from utils.paths import DATA_DIR, RAW_DIR

# sys.path.append('/Users/max/Deep_Sky')
# from creds import CDS_UID, CDS_API_key

logger = setup_logging()

class DatasetRetriever:
    def __init__(self, is_local=True):
        self.is_local = is_local
        self.logger = setup_logging(level='INFO')
        self.dataset_dir = load_config()
        
        try:
            if self.is_local:
                # In local environment, let cdsapi handle credentials from .cdsapirc
                self.client = cdsapi.Client()
                self.logger.info("Using local CDS credentials from .cdsapirc")
            else:
                # In Lambda, use credentials from Secrets Manager
                secrets = get_aws_secret()
                self.client = cdsapi.Client(
                    url=secrets['CDS_API_URL'],
                    key=secrets['CDS_API_KEY'],
                    quiet=True
                )
                self.logger.info("Successfully initialized CDS client with AWS credentials")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            raise

    def get_existing_file(self, dataset_name):
        """Get existing dataset file if it exists"""
        dataset_config = self.dataset_dir.get(dataset_name)
        if not dataset_config:
            self.logger.error(f"Dataset {dataset_name} not found in dataset_dir.json")
            return None
        
        var_name = dataset_config['era5_nc_variable']
        time_step = dataset_config['x_axis_unit']
        
        if self.is_local:
            pattern = RAW_DIR / f"{dataset_name}_*.nc"
            self.logger.info(f"Looking for files matching: {pattern}")
            files = glob.glob(str(pattern))
        else:
            # Use boto3 to list objects in S3
            s3 = boto3.client('s3')
            bucket = str(DATA_DIR)
            prefix = f'raw/{dataset_name}_'
            self.logger.info(f"Looking for files in S3 bucket: {bucket} with prefix: {prefix}")
            
            try:
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
                )
                files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.nc')]
            except Exception as e:
                self.logger.error(f"Error listing S3 objects: {e}")
                return None
        
        if len(files) == 1:
            self.logger.info(f"Found existing file: {files[0]}")
            try:
                if self.is_local:
                    with xr.open_dataset(files[0]) as file:
                        data = file.load()
                else:
                    # For S3, download to temporary file first
                    bucket, s3_key = self._normalize_s3_path(files[0])
                    temp_file = f'/tmp/{os.path.basename(s3_key)}'
                    s3.download_file(bucket, s3_key, temp_file)
                    with xr.open_dataset(temp_file) as file:
                        data = file.load()
                    os.remove(temp_file)  # Clean up temp file
                
                latest_date = pd.to_datetime(data.valid_time[-1].values).date()
                self.logger.info(f"Latest date in file: {latest_date}")
                return data
            except Exception as e:
                self.logger.error(f"Error reading netCDF file: {e}")
                return None
        elif len(files) > 1:
            self.logger.error(f"Multiple files found for {dataset_name}:")
            for f in files:
                self.logger.error(f"  - {f}")
        else:
            self.logger.info(f"No existing file found for {dataset_name}")
        return None

    def update_era5_dataset(self, dataset_name):
        """Update a specific dataset"""
        # Get the cdsapi logger directly from logging module
        cdsapi_logger = logging.getLogger('cdsapi')
        original_cdsapi_level = cdsapi_logger.level
        
        try:
            # Set loggers to ERROR level during data pull
            cdsapi_logger.setLevel(logging.ERROR)
            
            self.logger.info(f"\nProcessing dataset: {dataset_name}")
            
            dataset_config = self.dataset_dir[dataset_name]
            # Get existing data
            existing_file = self.get_existing_file(dataset_name)
            if existing_file is not None:
                start_date = pd.to_datetime(existing_file.valid_time[-1].values).date()
                self.logger.info(f"Will update from {start_date} to today")
            else:
                start_date = datetime.date(dataset_config['start_year'], 1, 1)
                self.logger.info(f"No existing file found. Will download all data from {start_date}")

            # Before setting end_date to today, add safety margin for data availability
            today = datetime.date.today()
            # ERA5 data typically has a 5-day delay
            safety_margin = datetime.timedelta(days=5)  
            available_end = today - safety_margin
            
            if start_date and start_date > available_end:
                self.logger.info(f"Dataset {dataset_name} is already up to date (latest available data)")
                return
            
            # Use the available_end instead of end_date if needed
            end_date = min(today, available_end)
            
            self.logger.info(f"Will update from {start_date} to {end_date}")

            # Pull new data
            self.logger.info(f"Pulling ERA5 data for {dataset_name}...")
            self._pull_era5_data(
                dataset_name=dataset_name,
                api_var=dataset_config['era5_api_variable'],
                nc_var=dataset_config['era5_nc_variable'],
                coords=dataset_config['era5_coords'],
                time_step=dataset_config['x_axis_unit'],
                existing_file=existing_file,
                start_date=start_date,
                end_date=end_date
            )

            self.logger.info(f"Successfully updated {dataset_name}")

        except Exception as e:
            self.logger.error(f"Error updating {dataset_name}: {str(e)}")
            raise
        
        finally:
            # Restore original logging levels
            cdsapi_logger.setLevel(original_cdsapi_level)

    def _pull_era5_data(self, dataset_name, api_var, nc_var, coords, time_step, existing_file, start_date, end_date):
        """Pull data from ERA5"""
        
        # Early detection of multi-month request
        spans_multiple_months = (time_step == 'month_day' and start_date.month != end_date.month)
        
        if spans_multiple_months:
            return self._handle_multi_month_request(
                dataset_name, api_var, nc_var, coords, time_step, 
                existing_file, start_date, end_date
            )
        
        # For single month or annual requests, proceed with normal flow
        temp_file = self._prepare_temp_files(dataset_name, end_date)
        
        try:
            # Prepare the API request
            request, endpoint = self._prepare_api_request(
                api_var, time_step, coords, start_date, end_date
            )
            
            # Retrieve the data
            self._retrieve_era5_data(endpoint, request, temp_file)
            
            # Process the retrieved data
            processed_data = self._process_era5_data(
                temp_file, dataset_name, nc_var, time_step
            )
            
            # Combine with existing data if available
            combined_data = self._combine_with_existing(
                processed_data, existing_file, time_step
            )
            
            # Save the output
            output_path = self._save_output(combined_data, dataset_name, end_date)
            
            return output_path
            
        finally:
            self._cleanup_temp_files(temp_file)
    
    def _handle_multi_month_request(self, dataset_name, api_var, nc_var, coords, 
                                   time_step, existing_file, start_date, end_date):
        """Handle requests that span multiple months by splitting them."""
        self.logger.info(f"Crossing month boundary: {start_date} to {end_date}")
        self.logger.info(f"Splitting request into two parts: {start_date} to end of month, and start of next month to {end_date}")
        
        # First request: from start_date to end of month
        last_day_of_month = pd.Timestamp(start_date.year, start_date.month, 1) + pd.offsets.MonthEnd(0)
        end_date_first_month = last_day_of_month.date()
        
        # Process first month
        first_output = self._pull_era5_data(
            dataset_name, api_var, nc_var, coords, time_step, 
            existing_file, start_date, end_date_first_month
        )
        
        # Process second month with properly resampled first month data
        second_existing_file = self._load_as_daily_means(first_output, time_step)
        
        # Then process second month
        start_date_second_month = pd.Timestamp(start_date.year, start_date.month, 1) + pd.offsets.MonthBegin(1)
        
        return self._pull_era5_data(
            dataset_name, api_var, nc_var, coords, time_step,
            second_existing_file, start_date_second_month.date(), end_date
        )
    
    def _load_as_daily_means(self, file_path, time_step):
        """Load a netCDF file and ensure it contains daily means."""
        if time_step != 'month_day':
            # For non-daily data, just load without resampling
            if self.is_local:
                with xr.open_dataset(file_path) as ds:
                    return ds.load()
            else:
                bucket, key = self._normalize_s3_path(file_path)
                temp_file = f'/tmp/temp_existing_file.nc'
                s3 = boto3.client('s3')
                s3.download_file(bucket, key, temp_file)
                with xr.open_dataset(temp_file) as ds:
                    result = ds.load()
                os.remove(temp_file)
                return result
        
        # For daily data, ensure we have daily means
        if self.is_local:
            with xr.open_dataset(file_path) as ds:
                return ds.resample(valid_time='1D').mean().load()
        else:
            bucket, key = self._normalize_s3_path(file_path)
            temp_file = f'/tmp/temp_existing_file.nc'
            s3 = boto3.client('s3')
            s3.download_file(bucket, key, temp_file)
            with xr.open_dataset(temp_file) as ds:
                result = ds.resample(valid_time='1D').mean().load()
            os.remove(temp_file)
            return result
    
    def _prepare_temp_files(self, dataset_name, end_date):
        """Prepare temporary and output file paths."""
        if self.is_local:
            temp_file = RAW_DIR / f'temp_{dataset_name}.nc'
        else:
            temp_file = f'/tmp/temp_{dataset_name}.nc'  # Use Lambda's temp directory
        
        return temp_file
    
    def _prepare_api_request(self, api_var, time_step, coords, start_date, end_date):
        """Prepare the API request based on time_step and date range."""
        if time_step == 'month_day':
            endpoint = "reanalysis-era5-single-levels"
            request = {
                'product_type': ['reanalysis'],
                'variable': api_var,
                'year': [str(start_date.year)],
                'month': [f'{start_date.month:02d}'],
                'day': [f'{i:02d}' for i in range(start_date.day, (end_date.day + 1))],
                'time': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'area': coords
            }
        else:  # year
            endpoint = "reanalysis-era5-single-levels-monthly-means"
            request = {
                'product_type': ['monthly_averaged_reanalysis'],
                'variable': api_var,
                'year': [str(i) for i in range(start_date.year, end_date.year + 1)],
                'month': [f'{i:02d}' for i in range(1, 13)],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'area': coords
            }
        
        self.logger.info(f"API request: {request}")
        return request, endpoint
    
    def _retrieve_era5_data(self, endpoint, request, temp_file):
        """Execute the ERA5 data retrieval."""
        # Add progress tracking
        self.logger.info(f"Submitting request to {endpoint}")
        
        # Add a timeout warning log
        self.logger.info(f"⚠️ CDS API calls can be slow. If running in Lambda, be aware of the 15-minute timeout.")
        
        # Add timestamp before starting the API call
        start_time = datetime.datetime.now()
        self.logger.info(f"Starting API call at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Actual data retrieval without progress callback
        self.client.retrieve(
            endpoint, 
            request, 
            str(temp_file)
        )
        
        # Log completion time and duration
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        self.logger.info(f"ERA5 data download completed in {duration.total_seconds()/60:.2f} minutes")
    
    def _process_era5_data(self, temp_file, dataset_name, nc_var, time_step):
        """Process the retrieved ERA5 data."""
        with xr.open_dataset(temp_file) as data:
            if time_step == 'month_day':
                # Log information about retrieved data
                formatted_times = [pd.to_datetime(t).strftime('%Y-%m-%d %H:%M') for t in data.valid_time.values]
                self.logger.info(f"Dates and times retrieved in latest pull: {formatted_times}")
                
                # Check for and handle incomplete days
                data = self._handle_incomplete_days(data, dataset_name)
                
                # Apply geographic weights and compute
                temp_mean = self._compute_weighted_mean(data, nc_var)
                
                # Always resample to daily means for consistent output
                if len(temp_mean.valid_time) > 0:
                    temp_mean = temp_mean.resample(valid_time='1D').mean()
            else:  # year
                self.logger.info("Annual data retrieval, no timeslice filtering needed.")
                
                # Apply geographic weights and compute
                temp_mean = self._compute_weighted_mean(data, nc_var)
                
                # Resample to yearly means
                temp_mean = temp_mean.chunk({'valid_time': -1})
                temp_mean = temp_mean.resample(valid_time='YS').mean()
            
            # Convert to dataset
            temp_mean_ds = temp_mean.to_dataset(name=nc_var)
            
            # Compute and return
            with ProgressBar():
                return temp_mean_ds.compute()
    
    def _handle_incomplete_days(self, data, dataset_name):
        """Filter out days with incomplete data (not 8 time slices)."""
        # Check for days with complete data (8 time slices)
        date_strings = [pd.to_datetime(t).strftime('%Y-%m-%d') for t in data.valid_time.values]
        date_counts = {}
        for date_str in date_strings:
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
        
        incomplete_days = [date for date, count in date_counts.items() if count != 8]
        
        # If there are incomplete days, handle them
        if incomplete_days:
            self.logger.warning(f"Days with incomplete data (not 8 time slices): {incomplete_days}")
            
            # Check if the last day is incomplete
            if incomplete_days[-1] == pd.to_datetime(data.valid_time.values[-1]).strftime('%Y-%m-%d'):
                self.logger.warning(f"Latest day has incomplete data and will be excluded")
            
            # Filter out incomplete days
            valid_times = []
            for time in data.valid_time.values:
                date_str = pd.to_datetime(time).strftime('%Y-%m-%d')
                if date_counts[date_str] == 8:
                    valid_times.append(True)
                else:
                    valid_times.append(False)
            
            data = data.isel(valid_time=valid_times)
            
            # Check if we have any data left after filtering
            if len(data.valid_time) > 0:
                self.logger.info(f"Filtered dataset to include only days with 8 time slices. New time range: {pd.to_datetime(data.valid_time.values[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(data.valid_time.values[-1]).strftime('%Y-%m-%d')}")
                data = data.resample(valid_time='1D').mean()
            else:
                self.logger.warning("No complete days found in the new data.")
                # Return empty dataset to signal no complete data
                return data
        else:
            self.logger.info("All days have complete data (8 time slices). Taking daily means.")
            data = data.resample(valid_time='1D').mean()
        
        return data
    
    def _compute_weighted_mean(self, data, nc_var):
        """Compute the weighted mean of the data."""
        temp_mean, _ = process_netcdf_data(
            ds=data,
            var_name=nc_var,
            weights=True,
            chunks={'valid_time': -1, 'latitude': 100, 'longitude': 100}
        )
        
        return temp_mean
    
    def _combine_with_existing(self, temp_mean_ds, existing_file, time_step):
        """Combine new data with existing data."""
        if existing_file is None:
            return temp_mean_ds
            
        # Drop expver if present
        if 'expver' in existing_file.coords:
            existing_file = existing_file.drop_vars('expver')
        if 'expver' in temp_mean_ds.coords:
            temp_mean_ds = temp_mean_ds.drop_vars('expver')
        
        # For month_day data, ensure both datasets are at daily resolution
        if time_step == 'month_day':
            existing_file, temp_mean_ds = self._ensure_daily_resolution(
                existing_file, temp_mean_ds
            )
        
        # Combine and remove duplicates
        combined_data = xr.concat([existing_file, temp_mean_ds], dim='valid_time') 
        combined_data = combined_data.sortby('valid_time').drop_duplicates('valid_time', keep='last')
        
        return combined_data
    
    def _ensure_daily_resolution(self, existing_file, new_data):
        """Ensure both existing and new datasets have daily resolution."""
        existing_times = pd.Series(existing_file.valid_time.values)
        new_times = pd.Series(new_data.valid_time.values)
        
        # Resample existing data if needed
        if len(existing_times) > 0 and len(existing_times.diff().dropna()) > 0:
            existing_freq_hours = existing_times.diff().dropna().median().total_seconds() / 3600
            if existing_freq_hours < 23:
                self.logger.info(f"Existing data has sub-daily frequency ({existing_freq_hours:.1f} hours). Resampling to daily.")
                existing_file = existing_file.resample(valid_time='1D').mean()
        
        # Resample new data if needed
        if len(new_times) > 0 and len(new_times.diff().dropna()) > 0:
            new_freq_hours = new_times.diff().dropna().median().total_seconds() / 3600
            if new_freq_hours < 23:
                self.logger.info(f"New data has sub-daily frequency ({new_freq_hours:.1f} hours). Resampling to daily.")
                new_data = new_data.resample(valid_time='1D').mean()
        
        return existing_file, new_data
    
    def _save_output(self, combined_data, dataset_name, end_date):
        """Save the output to file or S3."""
        if self.is_local:
            output_file = RAW_DIR / f'{dataset_name}_{end_date.strftime("%Y%m%d")}.nc'
            combined_data.to_netcdf(output_file)
            self.logger.info(f"Data written to {output_file}")
            
            cleanup_old_files(str(RAW_DIR / f'{dataset_name}_*.nc'), output_file)
            return str(output_file)
        else:
            # For S3 environment
            bucket = str(DATA_DIR)
            output_key = f'raw/{dataset_name}_{end_date.strftime("%Y%m%d")}.nc'
            output_file = f'/tmp/{dataset_name}_{end_date.strftime("%Y%m%d")}.nc'
            
            combined_data.to_netcdf(output_file)
            
            s3 = boto3.client('s3')
            with open(output_file, 'rb') as f:
                s3.put_object(
                    Bucket=bucket,
                    Key=output_key,
                    Body=f
                )
            self.logger.info(f"Data written to s3://{bucket}/{output_key}")
            
            cleanup_old_files(f'raw/{dataset_name}_*.nc', output_key, is_local=False)
            
            # Clean up local file
            if os.path.exists(output_file):
                os.remove(output_file)
                
            return f"s3://{bucket}/{output_key}"
    
    def _cleanup_temp_files(self, temp_file):
        """Clean up any temporary files."""
        if os.path.exists(temp_file):
            os.remove(temp_file)
            self.logger.info(f"Removed temporary file: {temp_file}")

    def _normalize_s3_path(self, path, bucket=None):
        """Normalize paths between local and S3 formats."""
        original_path = path
        if bucket is None:
            bucket = str(DATA_DIR)
            
        if path.startswith('s3://'):
            parts = path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        elif path.startswith('/tmp/'):
            key = f"raw/{os.path.basename(path)}"
        else:
            key = path
        
        self.logger.debug(f"Normalized path: '{original_path}' → bucket:'{bucket}', key:'{key}'")
        return bucket, key

def main():
    retriever = DatasetRetriever()
    era5_datasets = [name for name in retriever.dataset_dir.keys() if name.startswith('era5_')]

    for dataset in era5_datasets:
        try:
            retriever.update_era5_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to update {dataset}: {str(e)}")

if __name__ == '__main__':
    main()