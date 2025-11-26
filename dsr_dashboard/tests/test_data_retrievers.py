import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from utils.retrieval_utils import load_config
from retrieval.aviso_slr import download_aviso_slr
from retrieval.noaa_billion import download_noaa_billion
from retrieval.co2_ppm import download_co2_ppm
import urllib.error
import requests.exceptions

@pytest.fixture
def config():
    """Load real configuration from dataset_dir.json"""
    return load_config()

def test_aviso_real_data(config):
    """Integration test: Download and check real Aviso+ data"""
    df = download_aviso_slr(config)
    
    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert 'year' in df.columns
    assert 'msl' in df.columns
    
    # Check data types
    assert df['year'].dtype == 'datetime64[ns]'
    assert df['msl'].dtype == 'float64'
    
    # Check data quality
    assert len(df) > 300  # Should have many records
    assert df['year'].min().year >= 1993  # Data should start in 1993
    assert df['msl'].notna().all()  # No missing values
    assert df['msl'].between(-10, 200).all()  # Reasonable MSL (mm) range

@pytest.mark.integration
def test_noaa_real_data(config):
    """Integration test: Download and check real NOAA data"""
    try:
        df = download_noaa_billion(config)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        pytest.skip(f"NOAA API unavailable: {str(e)}")
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Network error accessing NOAA API: {str(e)}")
    
    # Only run these checks if we successfully got data
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['year', 'disaster_count', 'disaster_cost'])
    
    # Check data quality
    assert len(df) > 10  # Should have multiple years
    assert df['year'].min() >= 1980  # Data should go back to 1980
    assert df['disaster_count'].ge(0).all()  # Counts should be non-negative
    assert df['disaster_cost'].ge(0).all()  # Costs should be non-negative