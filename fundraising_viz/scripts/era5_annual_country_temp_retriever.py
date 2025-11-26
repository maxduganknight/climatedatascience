import sys
from pathlib import Path
dashboard_dir = Path(__file__).parent.parent.parent / 'dsr_dashboard'
sys.path.append(str(dashboard_dir))

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import datetime
import cdsapi
from shapely.geometry import Point
import os
import psutil

from utils.logging_utils import setup_logging
from utils.retrieval_utils import get_aws_secret

# Countries to process
COUNTRIES = [
    'United States of America',
    'Canada',
    'Norway',
    'Saudi Arabia',
    'United Kingdom',
    'Germany',
    'France',
    'Netherlands',
    'United Arab Emirates',
    'Qatar',
    'Singapore',
    'Japan',
    'Australia'
]

# Setup logging
logger = setup_logging()

class CountryTempProcessor:
    def __init__(self, countries=None):
        self.countries = countries or COUNTRIES
        self.logger = setup_logging(level='INFO')
        
        # Initialize CDS client
        try:
            self.client = cdsapi.Client()
            self.logger.info("Using local CDS credentials from .cdsapirc")

        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            raise
        
        # Load country boundaries
        self.country_gdf = self._load_country_boundaries()

    
    def _load_country_boundaries(self):
        """Load country boundaries from Natural Earth data"""
        self.logger.info("Loading country boundaries from Natural Earth...")
        
        # Use working Natural Earth source
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
        
        try:
            gdf = gpd.read_file(url)
            self.logger.info(f"Successfully loaded {len(gdf)} countries from Natural Earth")
        except Exception as e:
            self.logger.error(f"Could not load country boundaries from Natural Earth: {e}")
            raise Exception("Failed to load country boundaries - cannot proceed without proper geographic data")
        
        # Filter to requested countries using NAME column
        filtered_countries = []
        for country in self.countries:
            country_match = gdf[gdf['NAME'] == country]
            
            if len(country_match) > 0:
                filtered_countries.append(country_match)
                self.logger.info(f"Found boundary for {country}")
            else:
                self.logger.warning(f"Could not find boundary for {country}")
        
        if not filtered_countries:
            self.logger.error(f"Could not find boundaries for any requested countries: {self.countries}")
            raise Exception("No country boundaries found - cannot proceed")
        
        return pd.concat(filtered_countries, ignore_index=True)
    
    
    def retrieve_era5_data(self, start_year=1950, end_year=None):
        """Retrieve ERA5 temperature data for country analysis"""
        if end_year is None:
            # Don't go beyond 2024, and account for data lag
            current_year = datetime.date.today().year
            end_year = min(current_year - 1, 2024)
            
        self.logger.info(f"Retrieving ERA5 data from {start_year} to {end_year}")
        
        # Use global coordinates to match working dsr_dashboard format
        coords = [90, 180, -90, -180]  # Global: [North, East, South, West]
        
        # Prepare API request for annual data (working format confirmed by testing)
        request = {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': '2m_temperature',
            'year': [str(i) for i in range(start_year, end_year + 1)],
            'month': [f'{i:02d}' for i in range(1, 13)],
            'time': '00:00',
            'format': 'netcdf',
            'area': coords
        }
        
        # Download data
        temp_file = f'data/country_temp_anomalies/temp_era5_country_data_{datetime.date.today().strftime("%Y%m%d")}.nc'
        
        self.logger.info("Submitting request to CDS API...")
        self.logger.info(f"API request: {request}")
        start_time = datetime.datetime.now()
        
        self.client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            request,
            temp_file
        )
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Data download completed in {duration.total_seconds()/60:.2f} minutes")
        
        return temp_file
    
    def process_country_data(self, netcdf_file):
        """Process ERA5 data to extract country-specific temperature anomalies"""
        import gc
        
        self.logger.info("Processing ERA5 data for country analysis...")
        
        # Load the netCDF data with chunking to reduce memory usage
        with xr.open_dataset(netcdf_file, chunks={'valid_time': 12}) as ds:
            self.logger.info(f"Dataset dimensions: {ds.dims}")
            self.logger.info(f"Time range: {pd.to_datetime(ds.valid_time.values[0])} to {pd.to_datetime(ds.valid_time.values[-1])}")
            
            # Convert temperature from Kelvin to Celsius
            ds['t2m'] = ds['t2m'] - 273.15
            
            # Process data for each country
            country_results = {}
            
            for _, country_row in self.country_gdf.iterrows():
                country_name = country_row['NAME']
                country_geom = country_row['geometry']
                
                self.logger.info(f"Processing {country_name}...")
                
                # Create a mask for the country (memory-efficient approach)
                country_mask = self._create_country_mask(ds, country_geom)
                
                if country_mask.sum() == 0:
                    self.logger.warning(f"No grid points found for {country_name}")
                    continue
                
                # Extract temperature data for this country
                country_temp = ds['t2m'].where(country_mask)
                
                # Compute country-weighted mean
                country_mean = self._compute_country_mean(country_temp, ds)
                
                # Calculate annual means and anomalies
                annual_data = self._calculate_annual_anomalies(country_mean, country_name)
                
                country_results[country_name] = annual_data
                
                # Force cleanup after each country to free memory
                gc.collect()
            
            # Clean up temporary file
            if os.path.exists(netcdf_file):
                os.remove(netcdf_file)
                self.logger.info(f"Removed temporary file: {netcdf_file}")
            
            # Final cleanup
            gc.collect()
            return country_results
    
    def _create_country_mask(self, ds, country_geom):
        """Create a boolean mask for grid points within a country"""
        import gc
        
        # Get coordinate arrays
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        # First, use bounding box to limit the search area (much more efficient)
        bounds = country_geom.bounds  # (minx, miny, maxx, maxy)
        
        # Find lat/lon indices within bounding box (with small buffer)
        buffer = 1.0  # degrees
        lat_mask = (lats >= bounds[1] - buffer) & (lats <= bounds[3] + buffer)
        lon_mask = (lons >= bounds[0] - buffer) & (lons <= bounds[2] + buffer)
        
        # Get subset of coordinates
        lats_subset = lats[lat_mask]
        lons_subset = lons[lon_mask]
        
        if len(lats_subset) == 0 or len(lons_subset) == 0:
            self.logger.warning(f"No grid points found in country bounding box")
            # Return empty mask
            mask = np.zeros((len(lats), len(lons)), dtype=bool)
        else:
            # Create meshgrid for subset only (much smaller)
            lon_grid_subset, lat_grid_subset = np.meshgrid(lons_subset, lats_subset)
            
            # Use vectorized contains on the smaller subset
            try:
                try:
                    from shapely import contains_xy
                    mask_subset = contains_xy(country_geom, lon_grid_subset, lat_grid_subset)
                except ImportError:
                    from shapely.vectorized import contains
                    mask_subset = contains(country_geom, lon_grid_subset, lat_grid_subset)
            except Exception as e:
                self.logger.warning(f"Vectorized containment failed: {e}, using bounding box only")
                # Just use the bounding box as the mask
                mask_subset = np.ones_like(lon_grid_subset, dtype=bool)
            
            # Expand back to full grid
            mask = np.zeros((len(lats), len(lons)), dtype=bool)
            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]
            
            # Use numpy's advanced indexing to place the subset mask
            mask[np.ix_(lat_indices, lon_indices)] = mask_subset
            
            # Clean up subset variables
            del lon_grid_subset, lat_grid_subset, mask_subset
        
        # Force garbage collection
        gc.collect()
        
        # Convert to xarray DataArray with proper coordinates
        mask_da = xr.DataArray(
            mask,
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude']
        )
        
        return mask_da
    
    def _compute_country_mean(self, country_temp, ds):
        """Compute area-weighted mean temperature for country"""
        # Calculate weights based on latitude (cos(lat))
        weights = np.cos(np.deg2rad(ds.latitude))
        
        # Create a weighted dataset
        weights_da = xr.DataArray(weights, dims=['latitude'], coords={'latitude': ds.latitude})
        
        # Create mask for valid (non-NaN) points
        valid_mask = ~country_temp.isnull()
        
        # Apply weights only to valid country points
        weighted_temp = country_temp * weights_da
        weighted_area = valid_mask * weights_da
        
        # Compute weighted mean: sum of weighted values / sum of weights (for valid points only)
        country_mean = weighted_temp.sum(dim=['latitude', 'longitude']) / weighted_area.sum(dim=['latitude', 'longitude'])
        
        return country_mean
    
    def _calculate_annual_anomalies(self, temp_timeseries, country_name):
        """Calculate annual temperature anomalies from monthly data"""
        # Convert to pandas for easier handling
        df = temp_timeseries.to_dataframe(name='temperature').reset_index()
        df['year'] = pd.to_datetime(df['valid_time']).dt.year
        
        # Calculate annual means
        annual_means = df.groupby('year')['temperature'].mean().reset_index()
        
        # Calculate baseline (1991-2020)
        baseline_mask = (annual_means['year'] >= 1991) & (annual_means['year'] <= 2020)
        baseline_years = annual_means[baseline_mask]
        
        if len(baseline_years) == 0:
            self.logger.error("No 1991-2020 baseline data available. Cannot calculate anomalies without baseline period.")
            raise ValueError(f"Insufficient data for {country_name}: need data from 1991-2020 to calculate baseline")
        
        baseline_temp = baseline_years['temperature'].mean()
        self.logger.info(f"Calculated 1991-2020 baseline for {country_name}: {baseline_temp:.2f}°C")
        
        # Calculate anomalies
        annual_means['temperature_anomaly'] = annual_means['temperature'] - baseline_temp
        
        # Add pre-industrial adjustment using Copernicus methodology
        # Convert modern baseline to 1850-1900 pre-industrial baseline
        annual_means['temperature_anomaly_preindustrial'] = annual_means['temperature_anomaly'] + 0.88
        
        return annual_means.set_index('year')
    
    def save_results(self, country_results, output_dir='data/country_temp_anomalies/'):
        """Save country temperature results to separate CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving results to {output_path}")
        
        saved_files = []
        
        for country, data in country_results.items():
            # Create clean filename
            country_clean = country.lower().replace(' ', '_').replace('of_', '')
            filename = f'era5_{country_clean}_t2m_annual_anom.csv'
            filepath = output_path / filename
            
            # Prepare data for saving
            country_data = data.copy().reset_index()
            
            # Select and rename columns for output
            country_data = country_data.rename(columns={
                'temperature_anomaly': 'anom',
                'temperature_anomaly_preindustrial': 'anom_preindustrial'
            })
            
            # Reorder columns
            country_data = country_data[['year', 'temperature', 'anom', 'anom_preindustrial']]
            
            # Save to CSV
            country_data.to_csv(filepath, index=False)
            saved_files.append(str(filepath))
            self.logger.info(f"Saved {country} data to {filepath}")
        
        self.logger.info(f"All country files saved: {saved_files}")
        return saved_files
    
    def run_full_analysis(self, start_year=1950, end_year=None, output_dir='../data/global_temperatures/preprocessed_for_plots/'):
        """Run the complete country temperature analysis"""
        self.logger.info("Starting country temperature analysis...")
        
        # Step 1: Retrieve ERA5 data
        netcdf_file = self.retrieve_era5_data(start_year, end_year)
        
        # Step 2: Process country-specific data
        country_results = self.process_country_data(netcdf_file)
        
        # Step 3: Save results
        saved_files = self.save_results(country_results, output_dir)
        
        self.logger.info("Analysis complete!")
        return saved_files

def main():
    """Main execution function"""
    import sys
    import gc
    
    # Option to process one country at a time to avoid memory issues
    if len(sys.argv) > 1:
        # Single country mode: python script.py "Norway"
        single_country = sys.argv[1]
        if single_country in COUNTRIES:
            countries_to_process = [single_country]
            print(f"Processing single country: {single_country}")
        else:
            print(f"Error: '{single_country}' not in available countries: {COUNTRIES}")
            return
    else:
        # Process all countries (may require more memory)
        countries_to_process = COUNTRIES
        print(f"Processing all countries: {countries_to_process}")
    
    processor = None
    try:
        processor = CountryTempProcessor(countries=countries_to_process)
        
        saved_files = processor.run_full_analysis(
            start_year=1950,  # Reduced range to save memory
            end_year=None,  # Will use current year - 1
            output_dir='data/country_temp_anomalies/'
        )
        
        print(f"Analysis complete! Created {len(saved_files)} files:")
        for file in saved_files:
            print(f"  - {file}")
        
        if len(sys.argv) <= 1 and len(countries_to_process) > 1:
            print(f"\nTip: If memory issues occur, process countries individually:")
            for country in COUNTRIES:
                print(f"  python {sys.argv[0]} \"{country}\"")
                
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Cleanup to prevent semaphore leaks
        if processor is not None:
            del processor
        gc.collect()

if __name__ == '__main__':
    main()