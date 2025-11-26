import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import xarray as xr


def test_2024_air_temperature_raw():
    """Test that raw temperature values match climate reanalyzer.org data."""
    # Path to raw data
    raw_dir = Path("data/raw")
    nc_file = next(raw_dir.glob('era5_t2m_anom_month_day_*.nc'))
    
    
    # Check most recent file is not empty
    ds = xr.open_dataset(nc_file)
    ds_1979_01 = ds.sel(valid_time=ds.valid_time.dt.year.isin([1979]) &
                   ds.valid_time.dt.month.isin([1]))
    t2m_celsius = ds_1979_01.t2m.values - 273.15

    df_1979 = pd.DataFrame({
    'date': ds_1979_01.valid_time.values,
    't2m': t2m_celsius
    })

    mean_1979_01 = round(df_1979['t2m'].mean(), 4)

    assert mean_1979_01 == 12.0169, f"Mean temperature for January 1979 is not equal to expected value (°C). Expected: 12.0169. Got: {mean_1979_01}"
    
    # Check for expected values
    # assert df['year'].min() >= 1979, "Data should start in 1979"
    # assert df['t2m'].between(200, 320).all(), "Temperature values out of expected range"

def test_2024_air_temperature_anomaly():
    """Test that 2024 temperature anomaly exceeds expected threshold."""
    # Path to processed data
    processed_dir = Path("data/processed")
    annual_file = next(processed_dir.glob('era5_t2m_anom_year_*.csv'))
    month_day_file = next(processed_dir.glob('era5_t2m_anom_month_day_*.csv'))
    
    annual_df = pd.read_csv(annual_file)
    month_day_df = pd.read_csv(month_day_file)
    
    annual_2024_anomaly = annual_df[annual_df['year'] == 2024]['anom'].iloc[0]
    day_month_2024_anomaly = month_day_df['2024'].mean()

    assert day_month_2024_anomaly > 1.59, f"Month day 2024 temperature anomaly ({day_month_2024_anomaly:.2f}°C) is lower than expected (1.59°C)"
    assert annual_2024_anomaly > 1.59, f"Annual 2024 temperature anomaly ({annual_2024_anomaly:.2f}°C) is lower than expected (1.59°C)"

def test_data_completeness():
    """Test that processed data files exist and contain expected data."""
    processed_dir = Path("data/processed")
    
    # Check all expected files exist using glob patterns
    file_patterns = [
        'era5_t2m_anom_year_*.csv',
        'era5_t2m_anom_month_day_*.csv',
        'era5_sst_anom_month_day_*.csv',
        'co2_ppm_*.csv',
        'noaa_billion_*.csv',
        'aviso_slr_*.csv'
    ]
    
    for pattern in file_patterns:
        files = list(processed_dir.glob(pattern))
        assert len(files) > 0, f"No files found matching pattern: {pattern}"
        
        # Check most recent file is not empty
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_file)
        assert len(df) > 0, f"File is empty: {latest_file}"
        
        # Check required columns exist
        if 'month_day' in pattern:
            assert '2024' in df.columns, f"Missing 2024 column in {latest_file}"
        elif 'year' in pattern:
            required_cols = ['year', 'anom']
            assert all(col in df.columns for col in required_cols), f"Missing required columns in {latest_file}"
        elif 'aviso_slr' in pattern:
            required_cols = ['year', 'msl']
            assert all(col in df.columns for col in required_cols), f"Missing required columns in {latest_file}"

def test_anomaly_baseline():
    """Test that anomalies are calculated correctly relative to baseline period."""
    processed_dir = Path("data/processed")
    
    # Test annual temperature anomalies
    annual_file = next(processed_dir.glob('era5_t2m_anom_year_*.csv'))
    df = pd.read_csv(annual_file)
    
    # Check baseline period (1991-2020) has mean close to zero
    baseline_mask = (df['year'] >= 1991) & (df['year'] <= 2020)
    baseline_mean = df[baseline_mask]['anom'].mean()
    
    assert round(baseline_mean, 2) == 0.88, f"Baseline period mean ({baseline_mean:.3f}°C) is not close to pre-industrial baseline adjustment of 0.88°C"

def test_recent_warming_trend():
    """Test that recent years show expected warming trend."""
    processed_dir = Path("data/processed")
    annual_file = next(processed_dir.glob('era5_t2m_anom_year_*.csv'))
    df = pd.read_csv(annual_file)
    
    # Compare recent period (2014-2023) to previous decade (2004-2013)
    recent_mask = (df['year'] >= 2014) & (df['year'] <= 2023)
    previous_mask = (df['year'] >= 2004) & (df['year'] <= 2013)
    
    recent_mean = df[recent_mask]['anom'].mean()
    previous_mean = df[previous_mask]['anom'].mean()
    
    assert recent_mean > previous_mean, "Recent decade not warmer than previous decade"
    assert (recent_mean - previous_mean) > 0.2, "Warming trend weaker than expected"

def test_aviso_slr_data():
    """Test that Aviso+ SLR data is processed correctly."""
    processed_dir = Path("data/processed")
    file_pattern = 'aviso_slr_*.csv'
    files = list(processed_dir.glob(file_pattern))
    
    assert len(files) > 0, "No Aviso+ SLR files found"
    
    # Check most recent file is not empty
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    assert len(df) > 0, "Aviso+ SLR file is empty"
    
    # Check that sum of 1993 msl column is less than 3
    df['year'] = pd.to_datetime(df['year'])
    assert df[df['year'].dt.year == 1993]['msl'].sum() < 3, "Aviso+ SLR sum of 1993 msl column is greater than 3"