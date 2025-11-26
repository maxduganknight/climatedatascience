import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Union, Tuple
import logging
import warnings
import os
from .paths import LOGS_DIR, DATA_DIR
import boto3
from io import StringIO

def create_time_dataframe(values: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """
    Create DataFrame with time-based columns from raw values and times.
    
    Args:
        values: Array of temperature or other measurement values
        times: Array of corresponding timestamps
    
    Returns:
        DataFrame with time-based columns
    """
    df = pd.DataFrame({
        'value': values.flatten(),
        'time': pd.DatetimeIndex(times)
    })
    
    df['month_day'] = df['time'].dt.strftime('%m-%d')
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    
    return df

def calculate_baseline_values(
    df: pd.DataFrame, 
    start_year: int, 
    end_year: int, 
    group_col: str = 'month_day'
) -> pd.Series:
    """
    Calculate baseline values for a given period.
    
    Args:
        df: Input DataFrame with temperature data
        start_year: Start year for baseline period
        end_year: End year for baseline period
        group_col: Column to group by for baseline calculation
    
    Returns:
        Series of baseline values
    """
    baseline_period_mask = (df['year'] >= start_year) & (df['year'] <= end_year)
    return df[baseline_period_mask].groupby(group_col)['value'].mean()

def process_netcdf_data(
    ds: xr.Dataset,
    var_name: str,
    weights: bool = True,
    chunks: dict = None
) -> Tuple[xr.DataArray, float]:
    """
    Process NetCDF dataset with optional latitude weighting.
    
    Args:
        ds: Input xarray Dataset
        var_name: Variable name in dataset
        weights: Whether to apply latitude weighting
        chunks: Dictionary specifying chunk sizes for each dimension
    
    Returns:
        Tuple of (processed DataArray, mean value)
    """
    temp = ds[var_name]
    
    # Apply chunking if specified
    if chunks is not None:
        temp = temp.chunk(chunks)
    
    if weights:
        # Apply latitude weighting
        weights = np.cos(np.deg2rad(temp.latitude))
        weights.name = "weights"
        temp_weighted = temp.weighted(weights)
        temp_mean = temp_weighted.mean(["longitude", "latitude"])
    else:
        temp_mean = temp.mean(["longitude", "latitude"])
        
    return temp_mean, float(temp_mean.mean())
