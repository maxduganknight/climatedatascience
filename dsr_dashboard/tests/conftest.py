import warnings
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from selenium import webdriver

@pytest.fixture
def sample_era5_data():
    """Create a sample ERA5 dataset for testing."""
    # Create synthetic data
    times = pd.date_range('2020-01-01', '2022-12-31', freq='ME')
    lats = np.linspace(-90, 90, 73)
    lons = np.linspace(-180, 180, 144)
    
    # Create random temperature data
    temp_data = np.random.normal(15, 5, size=(len(times), len(lats), len(lons)))
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            't2m': (['time', 'latitude', 'longitude'], temp_data)
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        }
    )
    
    return ds

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory structure for testing."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir()
    return data_dir

@pytest.fixture(scope="session")
def webdriver_instance():
    """Create a shared webdriver instance for JavaScript tests"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

@pytest.fixture
def chart_test_data():
    """Create sample data for chart testing"""
    return {
        'bar': {
            'labels': ['2020', '2021', '2022', '2023'],
            'values': [10, 15, 20, 25]
        },
        'line': {
            'labels': ['Jan', 'Feb', 'Mar', 'Apr'],
            'values': [1, 2, 3, 4]
        },
        'daily_line': {
            'datasets': [
                {
                    'label': '2023',
                    'data': [{'x': '01-01', 'y': 1}, {'x': '01-02', 'y': 2}]
                },
                {
                    'label': '2024',
                    'data': [{'x': '01-01', 'y': 2}, {'x': '01-02', 'y': 3}]
                }
            ]
        }
    }