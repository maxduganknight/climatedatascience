import pytest
import pandas as pd
import numpy as np
import xarray as xr
from utils.era5_utils import create_time_dataframe, calculate_baseline_values, process_netcdf_data

def test_create_time_dataframe():
    """Test creation of time-based DataFrame."""
    # Create test data
    times = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    values = np.random.randn(len(times))
    
    # Create DataFrame
    df = create_time_dataframe(values, times)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['value', 'time', 'month_day', 'month', 'year'])
    assert len(df) == len(times)
    assert df['year'].unique() == [2020]
    assert set(df['month']) == set(range(1, 13))

def test_calculate_baseline_values():
    """Test baseline value calculations."""
    # Create test DataFrame
    df = pd.DataFrame({
        'year': [2019, 2019, 2020, 2020],
        'month_day': ['01-01', '02-01', '01-01', '02-01'],
        'value': [1.0, 2.0, 3.0, 4.0]
    })
    
    # Calculate baseline
    baseline = calculate_baseline_values(df, 2019, 2020)
    
    # Assertions
    assert isinstance(baseline, pd.Series)
    assert len(baseline) == 2  # Two unique month_days
    assert baseline['01-01'] == 2.0  # Mean of 1.0 and 3.0
    assert baseline['02-01'] == 3.0  # Mean of 2.0 and 4.0

def test_process_netcdf_data(sample_era5_data):
    """Test NetCDF data processing."""
    # Process data
    temp_mean, mean_value = process_netcdf_data(sample_era5_data, 't2m')
    
    # Assertions
    assert isinstance(temp_mean, xr.DataArray)
    assert isinstance(mean_value, float)
    assert temp_mean.dims == ('time',)  # Should only have time dimension after spatial averaging